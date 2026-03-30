"""Entry point for the minimal LLM test framework."""

from __future__ import annotations

import json

from llm_test_framework.core.runner import run_test


if __name__ == "__main__":
    demo_case = {
        "type": "basic",
        "prompt": "Write a function add(a, b)",
        "function_name": "add",
        "input": (2, 3),
        "expected": 5,
        "temperature": 0.2,
        "max_tokens": 128,
    }

    result = run_test("Salesforce/codegen-350M-mono", demo_case)
    print(json.dumps(result, indent=2, default=str))
