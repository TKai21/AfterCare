"""Run one of the evaluation scripts and print the model spend it caused. Dev only.

    .venv/bin/python scripts/run_with_spend.py scripts/eval_clone.py --repeats 1 --out-dir eval/clone/habits

The first argument is the script, the rest are its own arguments. Token counts
come from the `usage` of each model response (backend/llm.py); the cost is an
estimate from the prices below, which are assumptions, not a bill. The totals are
also appended to the file named by SPEND_LOG when it is set. Exits with the
script's own exit code.
"""
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

import llm  # noqa: E402

# US dollars per million tokens: (input, output).
PRICES = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
}


def estimate(usage: dict[str, dict[str, int]]) -> tuple[dict[str, float], list[str]]:
    """Cost per model, and the models that have no price (counted as 0)."""
    costs: dict[str, float] = {}
    unpriced: list[str] = []
    for model, entry in usage.items():
        price = PRICES.get(model)
        if price is None:
            unpriced.append(model)
            price = (0.0, 0.0)
        costs[model] = (entry["prompt_tokens"] * price[0] + entry["completion_tokens"] * price[1]) / 1_000_000
    return costs, unpriced


def report(usage: dict[str, dict[str, int]]) -> str:
    costs, unpriced = estimate(usage)
    lines = ["Model usage:"]
    for model, entry in sorted(usage.items()):
        lines.append(f"  {model}: {entry['calls']} calls, {entry['prompt_tokens']} input tokens, "
                     f"{entry['completion_tokens']} output tokens, about ${costs[model]:.4f}")
    lines.append(f"Estimated spend: ${sum(costs.values()):.4f}")
    if unpriced:
        lines.append(f"No price for: {', '.join(unpriced)} (counted as 0)")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: run_with_spend.py SCRIPT [ARGS...]", file=sys.stderr)
        return 2
    script = Path(argv[0]).resolve()
    sys.argv = [str(script), *argv[1:]]
    sys.path.insert(0, str(script.parent))
    llm.reset_usage()
    code = 0
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    finally:
        usage = llm.usage_snapshot()
        print(report(usage))
        log = os.environ.get("SPEND_LOG")
        if log:
            with open(log, "a") as fh:
                fh.write(json.dumps({"script": script.name, "usage": usage,
                                     "estimated_usd": sum(estimate(usage)[0].values())}) + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
