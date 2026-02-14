"""Cat judge: same evaluation logic and system prompt as cat project, self-contained in scoring."""

from .evaluator import evaluate_response
from .client import get_judge_client, LLMClient

__all__ = ["evaluate_response", "get_judge_client", "LLMClient"]
