from llm_test_framework.core.evaluator import evaluate_basic_case
from llm_test_framework.core.runner import run_test


class DummyTokenizer:
    eos_token_id = 0

    def __call__(self, prompt, return_tensors="pt"):
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    def decode(self, tokens, skip_special_tokens=True):
        return "def add(a, b):\n    return a + b"


class DummyModel:
    def generate(self, **kwargs):
        return [[1, 2, 3, 4]]


def test_basic_generation(monkeypatch):
    def fake_load_model(_model_name):
        return DummyModel(), DummyTokenizer()

    monkeypatch.setattr("llm_test_framework.core.runner.load_model", fake_load_model)

    case = {
        "type": "basic",
        "prompt": "Write a function add(a, b)",
        "function_name": "add",
        "input": (2, 3),
        "expected": 5,
    }
    result = run_test("dummy/model", case)

    assert result["generated_code"]
    assert "def" in result["generated_code"]


def test_execution():
    code = "def add(a, b):\n    return a + b"
    case = {
        "function_name": "add",
        "input": (2, 3),
        "expected": 5,
    }

    result = evaluate_basic_case(code, case)
    assert result["passed"] is True
