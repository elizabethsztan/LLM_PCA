import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from load_qwen import load_qwen_model

model, tokenizer = load_qwen_model()

# Storage for MLP inputs and outputs - using lists to accumulate data
mlp_activations = {
    'inputs': [],  # Changed to list to store all activations
    'outputs': []  # Changed to list to store all activations
}

def mlp_hook(module, input, output):
    # Input is a tuple, we take the first element
    mlp_activations['inputs'].append(input[0].detach().cpu())
    mlp_activations['outputs'].append(output.detach().cpu())
    print(f"Captured MLP activations - Input shape: {input[0].shape}, Output shape: {output.shape}")

# Run a forward pass to trigger the hook
test_prompts = ['Write a sentence that contains the word "cats".',
                'Write a sentence that doesnt contain the word “happy”',
                ]

# Register the hook on layer 19's MLP
layer_19_mlp = model.model.layers[19].mlp
hook_handle = layer_19_mlp.register_forward_hook(mlp_hook)

for prompt in test_prompts:

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0,
            # do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

# Remove the hook when done
hook_handle.remove()

# Save activations to a PyTorch file
print(f"\nTotal activations collected: {len(mlp_activations['inputs'])} forward passes")
print(f"Saving to mlp_activations.pt...")

torch.save({
    'inputs': mlp_activations['inputs'],
    'outputs': mlp_activations['outputs']
}, 'mlp_activations.pt')

print("Saved successfully!")
print("\nTo load later, use: data = torch.load('mlp_activations.pt')")