from llm_test_framework.core.evaluator import evaluate_humaneval_case, evaluate_pass_at_k


def test_humaneval_pass():
    prompt = "Write a function that returns True if a string is palindrome"
    samples = [
        "def solution(s):\n    return False",
        "def solution(s):\n    return s == s[::-1]",
        "def solution(s):\n    return len(s) > 0",
    ]
    hidden_tests = [
        (("aba",), True),
        (("abc",), False),
    ]

    result = evaluate_pass_at_k(prompt=prompt, generated_samples=samples, hidden_tests=hidden_tests)
    assert result["pass_at_k"] is True
    assert result["k"] == 3


def test_humaneval_fail():
    prompt = "Write a function that returns factorial of n"
    wrong_code = "def solution(n):\n    return n"
    hidden_tests = [
        ((5,), 120),
        ((3,), 6),
    ]

    result = evaluate_humaneval_case(prompt=prompt, generated_code=wrong_code, hidden_tests=hidden_tests)
    assert result["passed"] is False
