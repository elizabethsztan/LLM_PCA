import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import numpy as np
import pickle
import os

from symtorch import *

from load_qwen import load_qwen_model

#load the model
model, tokenizer = load_qwen_model()

# Load test prompts from JSON file
with open('train_prompts.json', 'r') as f:
    train_prompts = json.load(f)


#SETUP THE EXPERIMENT
results_location = "experiment3/setup0"
os.makedirs(results_location, exist_ok=True)

#which mlp layers to mess with
layers = [3]

#how many pca comps to use
pca_comps = 32


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
path = f"{results_location}/mlp_activations.pt"

if os.path.exists(path):
    print("Not running baseline - we already have the data we need")
    data = torch.load(path)
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


    # Save generated texts to JSON file
    with open(f'{results_location}/baseline_outputs.json', 'w') as f:
        json.dump(generated_texts, f, indent=2)

    print(f"Saved {len(generated_texts)} baseline outputs to baseline_outputs.json")


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

    torch.save(activations_to_save, f'{results_location}/mlp_activations.pt')

#Fit the PCA model if we haven't done it already

# Load the saved activations
data = torch.load(f"{results_location}/mlp_activations.pt")

# Dictionary to store PCA models for each layer
pca_models = {}

for layer in layers:
    pca_model_path = f'{results_location}/pca_models_{pca_comps}comps_layer{layer}.pkl'
    # Check if PCA model already exists
    if os.path.exists(pca_model_path):
        print(f"Loading existing PCA models from {pca_model_path}")
        with open(pca_model_path, 'rb') as f:
            pca_models[layer] = pickle.load(f)
        print(f"PCA loaded for layer {layer}.")
        print(f"  Inputs PCA explained variance: {pca_models[layer]['inputs'].explained_variance_ratio_.sum():.4f}")
        print(f"  Outputs PCA explained variance: {pca_models[layer]['outputs'].explained_variance_ratio_.sum():.4f}")
    else:
        print(f"Training the PCA for layer {layer}")

        #get inputs
        X_inputs = data[f'layer{layer}']['inputs'].numpy()

        pca_inputs = PCA(n_components=pca_comps, whiten=False, random_state=290402)
        pca_inputs.fit(X_inputs)

        print(f"Inputs PCA trained. Explained variance ratio sum: {pca_inputs.explained_variance_ratio_.sum():.4f}")

        #get outputs
        X_outputs = data[f'layer{layer}']['outputs'].numpy()

        pca_outputs = PCA(n_components=pca_comps, whiten=False, random_state=290402)
        pca_outputs.fit(X_outputs)

        print(f"Outputs PCA trained. Explained variance ratio sum: {pca_outputs.explained_variance_ratio_.sum():.4f}")

        # Save both input and output PCA models
        pca_models[layer] = {
            'inputs': pca_inputs,
            'outputs': pca_outputs
        }

        with open(pca_model_path, 'wb') as f:
            pickle.dump(pca_models[layer], f)
        print(f"PCA models saved to {pca_model_path}")

# Now fit a symbolic model with SymTorch

symbolic_models = {} #save the functions here

for layer in layers:
    symbolic_model_path = f'{results_location}/symbolic_model_{pca_comps}comps_layer{layer}'

    # Check if symbolic model already exists
    if os.path.exists(f'{symbolic_model_path}_metadata.pkl'):
        print(f"Loading existing symbolic model from {symbolic_model_path}")

        # Get the training data to create the callable function
        X_inputs = data[f'layer{layer}']['inputs'].numpy()
        X_outputs = data[f'layer{layer}']['outputs'].numpy()

        # Get PCA models for this layer
        pca_inputs = pca_models[layer]['inputs']
        pca_outputs = pca_models[layer]['outputs']

        # Transform data with PCA
        X_hat = pca_inputs.transform(X_inputs)
        Y_hat = pca_outputs.transform(X_outputs)

        # Create callable function for SymTorch
        def create_mapping_function(Y_hat_captured):
            """Create a function that maps reduced inputs to reduced outputs"""
            def f(X_reduced):
                # For now, return the captured Y_hat (will be replaced by symbolic regression)
                return Y_hat_captured
            return f

        f = create_mapping_function(Y_hat)

        # Load the symbolic model
        symbolic_model = SymbolicModel.load_model(symbolic_model_path, mlp_architecture=f)
        symbolic_model.switch_to_symbolic()

        symbolic_models[layer] = symbolic_model
        print(f"Symbolic model loaded for layer {layer}")

    else:
        print(f"Training symbolic model for layer {layer}")

        # Get the training data
        X_inputs = data[f'layer{layer}']['inputs'].numpy()
        X_outputs = data[f'layer{layer}']['outputs'].numpy()

        # Get PCA models for this layer
        pca_inputs = pca_models[layer]['inputs']
        pca_outputs = pca_models[layer]['outputs']

        # Transform data with PCA
        X_hat = pca_inputs.transform(X_inputs)
        Y_hat = pca_outputs.transform(X_outputs)

        # Create callable function for SymTorch
        def create_mapping_function(Y_hat_captured):
            """Create a function that maps reduced inputs to reduced outputs"""
            def f(X_reduced):
                # For now, return the captured Y_hat (will be replaced by symbolic regression)
                return Y_hat_captured
            return f

        f = create_mapping_function(Y_hat)

        # Create and train symbolic model
        symbolic_model = SymbolicModel(f, block_name=f'layer{layer}')
        symbolic_model.distill(X_hat) #run symbolic regression on this
        symbolic_model.switch_to_symbolic() #put in symbolic mode

        # Save the symbolic model
        symbolic_model.save_model(symbolic_model_path, save_pytorch=False, save_regressors=True)
        print(f"Symbolic model saved to {symbolic_model_path}")

        #save this symbolic model in the dictionary
        symbolic_models[layer] = symbolic_model



# Create intervention hooks for multiple layers
def make_mlp_intervention_hook(layer_num):
    """
    Create a hook function for PCA + symbolic intervention on a specific layer
    """
    def mlp_intervention_hook(module, input, output):
        # Store original device and dtype
        original_device = output.device
        original_dtype = output.dtype
        original_shape = output.shape

        # Get the MLP output (which is the input to our intervention)
        X_inputs = output.detach().cpu().numpy()
        X_inputs_reshaped = X_inputs.reshape(-1, X_inputs.shape[-1])

        # Apply input PCA to reduce dimensionality
        Z_inputs = pca_models[layer_num]['inputs'].transform(X_inputs_reshaped)

        # Apply symbolic model to get reduced output
        Z_outputs = symbolic_models[layer_num](Z_inputs)

        # Apply inverse output PCA to reconstruct full dimensionality
        X_hat = pca_models[layer_num]['outputs'].inverse_transform(Z_outputs)

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

# Remove all intervention hooks when done
for handle in intervention_handles:
    handle.remove()

# Save PCA-intervened generated texts to JSON file
layers_str = '_'.join(map(str, layers))
output_filename = f'{results_location}/pca_intervention_outputs_{pca_comps}comps_layers_{layers_str}.json'
with open(output_filename, 'w') as f:
    json.dump(generated_texts_pca, f, indent=2)

print(f"Saved {len(generated_texts_pca)} PCA intervention outputs to {output_filename}")
