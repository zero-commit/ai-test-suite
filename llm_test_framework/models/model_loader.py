"""Model loading utilities."""


def load_model(model_name: str):
    """Load a causal LM and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model id.

    Returns:
        Tuple of (model, tokenizer).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required to load models. Install with `pip install transformers`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer
