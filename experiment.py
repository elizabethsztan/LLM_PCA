import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import numpy as np
import pickle
import os

from load_qwen import load_qwen_model

model, tokenizer = load_qwen_model()
layer_19_mlp = model.model.layers[19].mlp
pca_comps = 0

# Load test prompts from JSON file
with open('test_prompts.json', 'r') as f:
    test_prompts = json.load(f)

# # Storage for MLP inputs and outputs - using lists to accumulate data
# mlp_activations = {
#     'inputs': [],  # Changed to list to store all activations
#     'outputs': []  # Changed to list to store all activations
# }

# def mlp_hook(module, input, output):
#     # Input is a tuple, we take the first element
#     mlp_activations['inputs'].append(input[0].detach().cpu())
#     mlp_activations['outputs'].append(output.detach().cpu())
#     # print(f"Captured MLP activations - Input shape: {input[0].shape}, Output shape: {output.shape}")




# # Register the hook on layer 19's MLP
# hook_handle = layer_19_mlp.register_forward_hook(mlp_hook)

# print("Running model no intervention")

# # Store all generated texts
# generated_texts = []

# for prompt in test_prompts:

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     with torch.no_grad():
#         generated_ids = model.generate(
#             **inputs,
#             max_new_tokens=50,
#             # temperature=0.0,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id
#         )

#     generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#     generated_texts.append({
#         'prompt': prompt,
#         'generated_text': generated_text
#     })

# # Remove the hook when done
# hook_handle.remove()

# # Save generated texts to JSON file
# with open('experiment_results/baseline_outputs.json', 'w') as f:
#     json.dump(generated_texts, f, indent=2)

# print(f"Saved {len(generated_texts)} baseline outputs to baseline_outputs.json")

# # Concatenate along the token dimension (assuming batch_size=1, we concatenate sequence lengths)
# inputs_tensor = torch.cat([x.squeeze(0) for x in mlp_activations['inputs']], dim=0)
# outputs_tensor = torch.cat([x.squeeze(0) for x in mlp_activations['outputs']], dim=0)

# print(f"Final shapes - Inputs: {inputs_tensor.shape}, Outputs: {outputs_tensor.shape}")
# print(f"Saving to mlp_activations.pt...")

# torch.save({
#     'inputs': inputs_tensor,
#     'outputs': outputs_tensor
# }, 'mlp_activations.pt')

data = torch.load('experiment_results/mlp_activations.pt')
outputs_tensor = data['outputs']

pca_model_path = f'pca_model_{pca_comps}comps.pkl'

# Check if PCA model already exists
if os.path.exists(pca_model_path):
    print(f"Loading existing PCA model from {pca_model_path}")
    with open(pca_model_path, 'rb') as f:
        pca = pickle.load(f)
    print(f"PCA loaded. Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
else:
    print("Training the PCA")

    X = outputs_tensor.numpy()
    print('shape of PCA training data:', X.shape)

    pca = PCA(n_components=pca_comps, whiten=False, random_state=290402)
    pca.fit(X)  # Fit PCA once on all collected activations

    print(f"PCA trained. Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")

    # Save the trained PCA model
    with open(pca_model_path, 'wb') as f:
        pickle.dump(pca, f)
    print(f"PCA model saved to {pca_model_path}")


def mlp_intervention_hook(module, input, output):
    """
    Hook for activation intervention experiments
    """
    # Store original device and dtype
    original_device = output.device
    original_dtype = output.dtype
    original_shape = output.shape

    # Transform to numpy for PCA
    X = output.detach().cpu().numpy()

    # Reshape for PCA if needed (batch_size * seq_len, hidden_dim)
    X_reshaped = X.reshape(-1, X.shape[-1])

    # Apply PCA transform and inverse (reconstruction)
    Z = pca.transform(X_reshaped)
    X_hat = pca.inverse_transform(Z)

    # Reshape back to original shape
    X_hat = X_hat.reshape(original_shape)

    # Convert back to torch tensor on the correct device with correct dtype
    X_hat_tensor = torch.from_numpy(X_hat).to(device=original_device, dtype=original_dtype)

    return X_hat_tensor 

intervention_hook_handle = layer_19_mlp.register_forward_hook(mlp_intervention_hook)

print("Running forward pass with PCA intervention")

# Store all PCA-intervened generated texts
generated_texts_pca = []

for prompt in test_prompts:

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids_pca = model.generate(
            **inputs,
            max_new_tokens=50,
            # temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False  # Disable KV caching so hook is called every forward pass
        )

    generated_text_pca = tokenizer.decode(generated_ids_pca[0], skip_special_tokens=True)
    generated_texts_pca.append({
        'prompt': prompt,
        'generated_text': generated_text_pca
    })

# Remove the hook when done
intervention_hook_handle.remove()

# Save PCA-intervened generated texts to JSON file
with open('experiment_results/pca_intervention_outputs.json', 'w') as f:
    json.dump(generated_texts_pca, f, indent=2)

print(f"Saved {len(generated_texts_pca)} PCA intervention outputs to pca_intervention_outputs.json")