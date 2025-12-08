"""
Script to load and interact with Qwen2.5-1.5B-Instruct model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_qwen_model(device="mps"):
    """
    Load Qwen2.5-1.5B-Instruct model and tokenizer

    Args:
        device: Device to load model on ("mps" for Mac GPU, "cpu", or "cuda")

    Returns:
        model, tokenizer
    """
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure loading options for Mac
    model_kwargs = {
        "torch_dtype": torch.float16 if device == "mps" else torch.float32,
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # Move to device
    model = model.to(device)

    print("Model loaded successfully!")
    print(f"Model device: {device}")
    print(f"Model dtype: {model.dtype}")

    return model, tokenizer
