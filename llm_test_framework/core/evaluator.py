"""Evaluation utilities for basic and HumanEval-style tests."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "str": str,
    "sum": sum,
}


def _safe_exec(code: str) -> Dict[str, Any]:
    namespace: Dict[str, Any] = {}
    exec(code, {"__builtins__": ALLOWED_BUILTINS}, namespace)
    return namespace


def evaluate_basic_case(generated_code: str, case: Dict[str, Any]) -> Dict[str, Any]:
    """Execute generated function and compare output with expected result."""
    try:
        namespace = _safe_exec(generated_code)
        fn = namespace[case["function_name"]]
        output = fn(*case["input"])
        passed = output == case["expected"]
        return {"passed": passed, "output": output, "expected": case["expected"]}
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def evaluate_humaneval_case(
    prompt: str,
    generated_code: str,
    hidden_tests: Iterable[Tuple[Tuple[Any, ...], Any]],
    function_name: str = "solution",
) -> Dict[str, Any]:
    """Evaluate one HumanEval-style candidate implementation."""
    _ = prompt
    try:
        namespace = _safe_exec(generated_code)
        fn = namespace[function_name]
        for args, expected in hidden_tests:
            value = fn(*args)
            if value != expected:
                return {"passed": False, "failed_input": args, "expected": expected, "actual": value}
        return {"passed": True}
    except Exception as exc:
        return {"passed": False, "error": str(exc)}


def evaluate_pass_at_k(
    prompt: str,
    generated_samples: List[str],
    hidden_tests: Iterable[Tuple[Tuple[Any, ...], Any]],
    function_name: str = "solution",
) -> Dict[str, Any]:
    """Simplified pass@k: True if any sample passes."""
    details = []
    for sample in generated_samples:
        result = evaluate_humaneval_case(
            prompt=prompt,
            generated_code=sample,
            hidden_tests=hidden_tests,
            function_name=function_name,
        )
        details.append(result)
        if result["passed"]:
            return {"pass_at_k": True, "k": len(generated_samples), "details": details}
    return {"pass_at_k": False, "k": len(generated_samples), "details": details}
