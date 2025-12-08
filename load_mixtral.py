"""
Script to load and interact with Mixtral-8x7B-v0.1 model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_mixtral_model(device="auto", load_in_8bit=False, load_in_4bit=False):
    """
    Load Mixtral-8x7B-v0.1 model and tokenizer

    Args:
        device: Device to load model on ("auto", "cuda", "cpu")
        load_in_8bit: Whether to load in 8-bit precision (requires bitsandbytes)
        load_in_4bit: Whether to load in 4-bit precision (requires bitsandbytes)

    Returns:
        model, tokenizer
    """
    model_name = "mistralai/Mixtral-8x7B-v0.1"

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model {model_name}...")
    print(f"Note: This is a large model (47B parameters). Loading may take time and require significant GPU memory.")

    # Configure loading options
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": device,
    }

    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        print("Loading in 8-bit mode...")
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
        print("Loading in 4-bit mode...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    print("Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Model dtype: {model.dtype}")

    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    """
    Generate text using the model

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    # Load the model
    # For large models like Mixtral, you may want to use quantization:
    # - Use load_in_8bit=True for 8-bit quantization
    # - Use load_in_4bit=True for 4-bit quantization (requires less memory)

    model, tokenizer = load_mixtral_model(
        device="auto",
        load_in_4bit=True  # Recommended for better memory efficiency
    )

    # Example usage
    prompt = "Explain what a Mixture of Experts model is:"
    print(f"\nPrompt: {prompt}")
    print("\nGenerating response...\n")

    response = generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=200,
        temperature=0.7
    )

    print(response)

    # Interactive mode
    print("\n" + "="*50)
    print("Interactive mode - Enter prompts (or 'quit' to exit)")
    print("="*50)

    while True:
        user_prompt = input("\nYour prompt: ")
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            break

        response = generate_text(
            model,
            tokenizer,
            user_prompt,
            max_new_tokens=200
        )
        print(f"\nResponse:\n{response}")
