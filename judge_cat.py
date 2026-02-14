"""
CLI to evaluate experiment results with a judge.
  python judge_cat.py <experiment> <judge>
- Loads results/<experiment>/results/*.json (format: [{ task, query, response }, ...]).
- Runs the given judge and saves to results/<experiment>/eval_by_<judge>/*.json
  (format: num_tasks, num_success, attack_success_rate, results).
- Implemented judges: cat (crescendo etc. to be added).
"""
import json
import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional

# Load .env from scoring directory so OPENAI_API_KEY is available
try:
    from dotenv import load_dotenv
    _scoring_dir = Path(__file__).resolve().parent
    load_dotenv(_scoring_dir / ".env")
except ImportError:
    pass

from judge_cat import evaluate_response, get_judge_client, LLMClient


def load_result_items(path: Path) -> List[Dict[str, Any]]:
    """Load list of result items from JSON (supports array or { results: [] } format)."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    raise ValueError(
        f"Unexpected JSON format in {path}: expected list or dict with 'results' key"
    )


def evaluate_one(
    task: int,
    query: str,
    response: str,
    judge_llm_client: LLMClient,
) -> Dict[str, Any]:
    """Evaluate one (task, query, response) with judge; return dict with score and success."""
    evaluation = evaluate_response(
        user_prompt=query,
        agent_response=response,
        judge_llm_client=judge_llm_client,
    )
    return {
        "score": evaluation["score"],
        "success": evaluation["success"],
    }


def run_judge_on_file(
    input_path: Path,
    output_path: Path,
    judge_model: str,
    limit: Optional[int] = None,
    print_lock: Optional[threading.Lock] = None,
) -> None:
    """Load one result file, judge each item, write to output_path. Uses its own judge client (safe for parallel)."""
    judge_llm_client = get_judge_client(judge_model)
    items = load_result_items(input_path)
    if limit is not None:
        items = items[:limit]

    prefix = f"[{input_path.name}] " if print_lock else ""
    def log(msg: str) -> None:
        if print_lock:
            with print_lock:
                print(prefix + msg)
        else:
            print(msg)

    results_out = []
    total = len(items)
    for i, item in enumerate(items, 1):
        task_id = item.get("task", i)
        query = item.get("query", "")
        response = item.get("response", "")
        if not query or not response:
            log(f"  [{i}/{total}] task={task_id} skip: missing query or response")
            results_out.append({
                "task": task_id,
                "query": query,
                "response": response,
                "score": 0.0,
                "success": False,
            })
            continue
        log(f"  [{i}/{total}] Evaluating task {task_id}...")
        try:
            ev = evaluate_one(task_id, query, response, judge_llm_client)
            results_out.append({
                "task": task_id,
                "query": query,
                "response": response,
                "score": ev["score"],
                "success": ev["success"],
            })
            status = "SUCCESS" if ev["success"] else "FAILED"
            log(f"  [{i}/{total}] task {task_id} score={ev['score']:.3f} {status}")
        except Exception as e:
            log(f"  [{i}/{total}] ERROR: {e}")
            results_out.append({
                "task": task_id,
                "query": query,
                "response": response,
                "score": 0.0,
                "success": False,
            })

    num_tasks = len(results_out)
    num_success = sum(1 for r in results_out if r.get("success") is True)
    rate = (num_success / num_tasks) if num_tasks else 0.0

    out = {
        "num_tasks": num_tasks,
        "num_success": num_success,
        "attack_success_rate": round(rate, 4),
        "results": results_out,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log(f"  -> {output_path} (num_tasks={num_tasks}, num_success={num_success}, rate={rate:.4f})")


# Judge name -> implementation (register new judges here)
AVAILABLE_JUDGES = {"cat"}  # crescendo planned


def main():
    parser = argparse.ArgumentParser(
        description="Load results/<experiment>/results/*.json, run judge, save to results/<experiment>/eval_by_<judge>/"
    )
    parser.add_argument(
        "experiment",
        type=str,
        help="Experiment (attack) name. JSONs are loaded from results/<experiment>/results/",
    )
    parser.add_argument(
        "judge",
        type=str,
        help="Judge name. Output is saved to results/<experiment>/eval_by_<judge>/ (e.g. cat, crescendo)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-4o-mini",
        help="Judge LLM model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of tasks per file (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of files to process in parallel (default: 1 = sequential)",
    )
    args = parser.parse_args()

    if args.judge not in AVAILABLE_JUDGES:
        print(f"Judge '{args.judge}' is not implemented yet. Available: {sorted(AVAILABLE_JUDGES)}")
        sys.exit(1)

    scoring_dir = Path(__file__).resolve().parent
    input_dir = scoring_dir / "results" / args.experiment / "results"
    output_dir = scoring_dir / "results" / args.experiment / f"eval_by_{args.judge}"

    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}")
        sys.exit(1)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No .json files in {input_dir}")
        sys.exit(0)

    print(f"Judge: {args.judge} (model: {args.judge_model})")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files to process: {len(json_files)}")
    if args.limit:
        print(f"Limit: {args.limit} tasks per file")
    if args.workers > 1:
        print(f"Workers: {args.workers} (parallel)")

    print_lock = threading.Lock() if args.workers > 1 else None
    max_workers = min(args.workers, len(json_files))

    if max_workers <= 1:
        for jf in json_files:
            print(f"\n--- {jf.name} ---")
            out_path = output_dir / jf.name
            try:
                run_judge_on_file(jf, out_path, args.judge_model, limit=args.limit, print_lock=None)
            except Exception as e:
                print(f"  ERROR: {e}")
                raise
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_judge_on_file,
                    jf,
                    output_dir / jf.name,
                    args.judge_model,
                    args.limit,
                    print_lock,
                ): jf
                for jf in json_files
            }
            for future in as_completed(futures):
                jf = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"  [{jf.name}] ERROR: {e}")
                    raise

    print(f"\nDone. Results saved under {output_dir}")


if __name__ == "__main__":
    main()
