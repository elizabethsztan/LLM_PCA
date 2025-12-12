import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import numpy as np
import pickle
import os
import math

from load_qwen import load_qwen_model

#load the model
model, tokenizer = load_qwen_model()

# Load prompts from JSON file
with open('train_prompts.json', 'r') as f:
    train_prompts = json.load(f)

with open('test_prompts.json', 'r') as f:
    test_prompts = json.load(f)

with open('harry_potter.json', 'r') as f:
    perplexity_text = json.load(f)


#SETUP THE EXPERIMENT
base_experiment_folder = "experiment2"
results_location = "experiment2/setup4"
os.makedirs(base_experiment_folder, exist_ok=True)
os.makedirs(results_location, exist_ok=True)

#which mlp layers to mess with
layers = [3, 7, 11, 15, 19, 23, 27]

#how many pca comps to use
pca_comps = 16


#create storage for MLP I/O
mlp_activations = {}
handles = []


def make_mlp_hook(layer_num):
    """Create a hook function for a specific layer"""
    def mlp_hook(module, input, output):
        mlp_activations[f"layer{layer_num}"]['inputs'].append(input[0].detach().cpu())
        mlp_activations[f"layer{layer_num}"]['outputs'].append(output.detach().cpu())
    return mlp_hook

# Only run below code if we don't have the information already
# Store baseline data in the base experiment folder (shared across all setups)
path = f"{base_experiment_folder}/mlp_activations.pt"

if os.path.exists(path):
    print("Not running baseline - we already have the data we need")
else:
    print("Running baseline and collecting activations")

    for layer in layers:
        # Create storage
        mlp_activations[f"layer{layer}"] = {'inputs': [], 'outputs': []}
        
        #register hook
        handle = model.model.layers[layer].mlp.register_forward_hook(make_mlp_hook(layer))
        handles.append(handle)

    print("Running model no intervention")

    # Store all generated texts
    generated_texts = []

    for prompt in train_prompts:

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                # temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_texts.append({
            'prompt': prompt,
            'generated_text': generated_text
        })


    # Remove the hook when done
    for handle in handles:
        handle.remove()


    # Save generated texts to JSON file in base experiment folder
    with open(f'{base_experiment_folder}/baseline_outputs.json', 'w') as f:
        json.dump(generated_texts, f, indent=2)

    print(f"Saved {len(generated_texts)} baseline outputs to {base_experiment_folder}/baseline_outputs.json")

    generated_texts_test = []
    #run a baseline test for the test prompts too
    for prompt in test_prompts:

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                # temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text_test = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_texts_test.append({
            'prompt': prompt,
            'generated_text': generated_text_test
        })

    # Save generated texts to JSON file in base experiment folder
    with open(f'{base_experiment_folder}/test_baseline_outputs.json', 'w') as f:
        json.dump(generated_texts_test, f, indent=2)

    print(f"Saved {len(generated_texts_test)} test baseline outputs to {base_experiment_folder}/test_baseline_outputs.json")


    # Save activations for each layer separately
    activations_to_save = {}
    for layer in layers:
        # Concatenate along the token dimension (assuming batch_size=1, we concatenate sequence lengths)
        inputs_tensor = torch.cat([x.squeeze(0) for x in mlp_activations[f'layer{layer}']['inputs']], dim=0)
        outputs_tensor = torch.cat([x.squeeze(0) for x in mlp_activations[f'layer{layer}']['outputs']], dim=0)

        activations_to_save[f'layer{layer}'] = {
            'inputs': inputs_tensor,
            'outputs': outputs_tensor
        }
        print(f"Layer {layer} - Inputs: {inputs_tensor.shape}, Outputs: {outputs_tensor.shape}")

    torch.save(activations_to_save, f'{base_experiment_folder}/mlp_activations.pt')

# Run the test for baseline perplexity
# Store in base experiment folder (shared across all setups)
path_perp = f"{base_experiment_folder}/baseline_perplexity.json"

if os.path.exists(path_perp):
    print("Not running baseline perplexity test - we already have the data we need")
    with open(path_perp, 'r') as f:
        baseline_perp_data = json.load(f)
        perplexity = baseline_perp_data['perplexity']
else:
    inputs = tokenizer(perplexity_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            labels=inputs['input_ids']
        )

        # Get the average negative log-likelihood per token
        nll = outputs.loss.item()

    # Perplexity = exp(NLL)
    perplexity = math.exp(nll)

    # Save baseline perplexity in base experiment folder
    baseline_perp_data = {
        'perplexity': perplexity,
        'nll': nll
    }
    with open(path_perp, 'w') as f:
        json.dump(baseline_perp_data, f, indent=2)
    print(f"Saved baseline perplexity: {perplexity:.4f}")


#Fit the PCA model if we haven't done it already

# Load the saved activations from base experiment folder
data = torch.load(f"{base_experiment_folder}/mlp_activations.pt")

# Dictionary to store PCA models for each layer
pca_models = {}

for layer in layers:

    pca_model_path = f'{results_location}/pca_model_{pca_comps}comps_layer{layer}.pkl'
    # Check if PCA model already exists
    if os.path.exists(pca_model_path):
        print(f"Loading existing PCA model from {pca_model_path}")
        with open(pca_model_path, 'rb') as f:
            pca_models[layer] = pickle.load(f)
        print(f"PCA loaded for layer {layer}. Explained variance ratio sum: {pca_models[layer].explained_variance_ratio_.sum():.4f}")
    else:
        print(f"Training the PCA for layer {layer}")

        # Get the outputs for this specific layer
        X = data[f'layer{layer}']['outputs'].numpy()
        print(f'Layer {layer} - shape of PCA training data:', X.shape)

        pca = PCA(n_components=pca_comps, whiten=False, random_state=290402)
        pca.fit(X)  # Fit PCA once on all collected activations

        print(f"PCA trained. Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")

        # Save the trained PCA model
        with open(pca_model_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"PCA model saved to {pca_model_path}")

        pca_models[layer] = pca

# Create intervention hooks for multiple layers
def make_mlp_intervention_hook(layer_num):
    """
    Create a hook function for PCA intervention on a specific layer
    """
    def mlp_intervention_hook(module, input, output):
        # Store original device and dtype
        original_device = output.device
        original_dtype = output.dtype
        original_shape = output.shape

        # Transform to numpy for PCA
        X = output.detach().cpu().numpy()

        # Reshape for PCA if needed (batch_size * seq_len, hidden_dim)
        X_reshaped = X.reshape(-1, X.shape[-1])

        # Apply PCA transform and inverse (reconstruction)
        Z = pca_models[layer_num].transform(X_reshaped)
        X_hat = pca_models[layer_num].inverse_transform(Z)

        # Reshape back to original shape
        X_hat = X_hat.reshape(original_shape)

        # Convert back to torch tensor on the correct device with correct dtype
        X_hat_tensor = torch.from_numpy(X_hat).to(device=original_device, dtype=original_dtype)

        return X_hat_tensor
    return mlp_intervention_hook

# Register intervention hooks for all layers
intervention_handles = []
for layer in layers:
    handle = model.model.layers[layer].mlp.register_forward_hook(make_mlp_intervention_hook(layer))
    intervention_handles.append(handle)

print("Running forward pass with PCA intervention")

# Store all PCA-intervened generated texts
generated_texts_pca = []
generated_texts_pca_test = []

for prompt in train_prompts:

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

    generated_text_pca_test = tokenizer.decode(generated_ids_pca[0], skip_special_tokens=True)
    generated_texts_pca_test.append({
        'prompt': prompt,
        'generated_text': generated_text_pca_test
    })

# Test perplexity of harry potter text with PCA intervention
inputs = tokenizer(perplexity_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(
        **inputs,
        labels=inputs['input_ids'],
        use_cache=False
    )

    # Get the average negative log-likelihood per token
    nll_pca = outputs.loss.item()

# Perplexity = exp(NLL)
perplexity_pca = math.exp(nll_pca)

# Remove all intervention hooks when done
for handle in intervention_handles:
    handle.remove()

# Save PCA-intervened generated texts to JSON file
layers_str = '_'.join(map(str, layers))
output_filename_train = f'{results_location}/pca_intervention_outputs_{pca_comps}comps_layers_{layers_str}.json'
with open(output_filename_train, 'w') as f:
    json.dump(generated_texts_pca, f, indent=2)

print(f"Saved {len(generated_texts_pca)} PCA intervention outputs (train set) to {output_filename_train}")

output_filename_test = f'{results_location}/pca_intervention_outputs_{pca_comps}comps_layers_{layers_str}_test.json'
with open(output_filename_test, 'w') as f:
    json.dump(generated_texts_pca_test, f, indent=2)

print(f"Saved {len(generated_texts_pca_test)} PCA intervention outputs (test set) to {output_filename_test}")

# Save results summary
results_summary = {
    'baseline_perplexity': perplexity,
    'pca_intervention_perplexity': perplexity_pca,
    'pca_components': pca_comps,
    'layers_intervened': layers,
    'perplexity_change': perplexity_pca - perplexity,
    'perplexity_ratio': perplexity_pca / perplexity
}

results_filename = f'{results_location}/results_summary.json'
with open(results_filename, 'w') as f:
    json.dump(results_summary, f, indent=2)

print(f"\nResults Summary:")
print(f"  Baseline Perplexity: {perplexity:.4f}")
print(f"  PCA Intervention Perplexity: {perplexity_pca:.4f}")
print(f"  Perplexity Change: {perplexity_pca - perplexity:.4f}")
print(f"  Perplexity Ratio: {perplexity_pca / perplexity:.4f}")
print(f"  PCA Components: {pca_comps}")
print(f"  Layers: {layers}")
print(f"\nSaved results summary to {results_filename}")