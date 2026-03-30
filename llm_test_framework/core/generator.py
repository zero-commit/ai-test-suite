"""Generation utilities."""


def generate_code(model, tokenizer, prompt: str, temperature: float = 0.2, max_tokens: int = 128) -> str:
    """Generate code from a prompt.

    Args:
        model: Loaded HF model.
        tokenizer: Loaded HF tokenizer.
        prompt: Natural language prompt.
        temperature: Sampling temperature.
        max_tokens: Maximum new tokens to generate.
    """
    encoded = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **encoded,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
