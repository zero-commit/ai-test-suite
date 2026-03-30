"""Test runner orchestration."""

from llm_test_framework.core.evaluator import evaluate_basic_case, evaluate_humaneval_case, evaluate_pass_at_k
from llm_test_framework.core.generator import generate_code
from llm_test_framework.models.model_loader import load_model


def run_test(model_name: str, test_case: dict) -> dict:
    """Run one test case end-to-end.

    test_case supports:
      - type=basic
      - type=humaneval
      - type=humaneval_passk
    """
    model, tokenizer = load_model(model_name)

    case_type = test_case.get("type", "basic")
    temperature = test_case.get("temperature", 0.2)
    max_tokens = test_case.get("max_tokens", 128)

    if case_type == "humaneval_passk":
        k = test_case.get("k", 3)
        samples = [
            generate_code(model, tokenizer, test_case["prompt"], temperature=temperature, max_tokens=max_tokens)
            for _ in range(k)
        ]
        result = evaluate_pass_at_k(
            prompt=test_case["prompt"],
            generated_samples=samples,
            hidden_tests=test_case["hidden_tests"],
            function_name=test_case.get("function_name", "solution"),
        )
        result["generated_samples"] = samples
        return result

    generated_code = generate_code(
        model=model,
        tokenizer=tokenizer,
        prompt=test_case["prompt"],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if case_type == "humaneval":
        result = evaluate_humaneval_case(
            prompt=test_case["prompt"],
            generated_code=generated_code,
            hidden_tests=test_case["hidden_tests"],
            function_name=test_case.get("function_name", "solution"),
        )
    else:
        result = evaluate_basic_case(generated_code, test_case)

    result["generated_code"] = generated_code
    return result
