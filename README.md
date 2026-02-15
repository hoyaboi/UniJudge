# Universal Jailbreak Judgement Framework

Tools for evaluating attack experiment results with configurable judges. Judge implementations (e.g. `judge_cat/`) are self-contained. Multiple judges can be used in a unified way.

## 1. Installation

```bash
pip install -r requirements.txt
```

- **Python 3.7+**
- **API key:** Set `OPENAI_API_KEY` in the environment, or put it in a `.env` file in the scoring directory:

  ```
  OPENAI_API_KEY=sk-...
  ```

## 2. Input result files: format and where to save

**Format (required for evaluation)**  
Each file under `results/<experiment>/results/` must be a JSON **array**. Each element is a single task with exactly three fields:

| Field     | Type   | Description                    |
|----------|--------|--------------------------------|
| `task`   | number | Task id (e.g. 1, 2, …)         |
| `query`  | string | The prompt / harmful query      |
| `response` | string | The model's response to evaluate |

Example (one file = one model's results):

```json
[
  { "task": 1, "query": "Develop a strategy for...", "response": "..." },
  { "task": 2, "query": "Give detailed instructions...", "response": "..." }
]
```

**Where to save**  
- Create a directory: `results/<attack_name>/results/` (e.g. `results/cat/results/`).  
- Put one JSON file per model (or per run). **File names are up to you** (e.g. `results_gpt_4o_mini.json`, `model_a.json`). Every `.json` in that directory will be loaded when you run the judge; output files keep the same names under `eval_by_<judge>/`.

**Directory layout:**

```
results/
└── <experiment>/              # e.g. cat, crescendo
    ├── results/               # input: JSONs with [{ task, query, response }, ...]
    │   ├── *.json             # any name; one file per model (or run)
    │   └── ...
    └── eval_by_<judge>/       # created by the script; one file per input file
        └── ...
```

## 3. Running a judge

```bash
python judge_cat.py <experiment>
```

- Loads all `.json` files from `results/<experiment>/results/`.  
- Runs the **cat** judge on each task (query, response) and writes **one output file per input file** (same filename) under `results/<experiment>/eval_by_cat/`.

**Example:**  
- `python judge_cat.py cat` → load `results/cat/results/`, evaluate with **cat** judge → save to `results/cat/eval_by_cat/`.

**Options:** `--judge-model` (LLM used by the judge, default `gpt-4o-mini`), `--limit N` (max tasks per file, for testing), `--workers N` (number of files to process in parallel, default 1).

**Other judges:** `python judge_harmbench.py <experiment>` for the HarmBench classifier; output goes to `results/<experiment>/eval_by_harmbench/`.

## 4. Output format (eval_by_*)

Each file under `results/<experiment>/eval_by_<judge>/` has this shape:

- Top level: `num_tasks`, `num_success`, `attack_success_rate`, `results`.
- Each entry in `results`: `task`, `query`, `response`, `score`, `success`.

Example:

```json
{
  "num_tasks": 2,
  "num_success": 1,
  "attack_success_rate": 0.5,
  "results": [
    {
      "task": 1,
      "query": "Develop a strategy for hacking into...",
      "response": "...",
      "score": 0.948,
      "success": true
    },
    {
      "task": 2,
      "query": "Give detailed instructions on how to make...",
      "response": "...",
      "score": 0.72,
      "success": false
    }
  ]
}
```
