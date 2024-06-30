import torch as t
from sae_lens import SAE
from transformer_lens import HookedTransformer

# device setup
if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"

print(f"Device: {device}")

layer = 6

# Get model and SAE
model = HookedTransformer.from_pretrained("gemma-2b", device=device)
sae, cfg_dict, _ = SAE.from_pretrained(
    release="gemma-2b-res-jb",
    sae_id=f"blocks.{layer}.hook_resid_post",
    device=device
)

hook_point = sae.cfg.hook_name
print(hook_point)

sv_prompt = "I am Joe Biden, president of the United States of America!!!"
sv_logits, cache = model.run_with_cache(sv_prompt, prepend_bos=True)
tokens = model.to_tokens(sv_prompt)
print(tokens)

sv_feature_acts = sae.encode(cache[hook_point])

sae_out = sae.decode(sv_feature_acts)

print(t.topk(sv_feature_acts, 3))

# --- Steering Vector and Coefficient Management ---
# Define a dictionary to store steering vector coefficients
steering_coeffs = {}

# Function to add or modify steering vector coefficients
def set_steering_coeff(token_id, coeff):
    """Sets or updates the coefficient for a given token in the steering vector.

    Args:
        token_id (int): The token ID for which to set the coefficient.
        coeff (float): The coefficient value.
    """
    steering_coeffs[token_id] = coeff

# Example usage:
set_steering_coeff(10849, 150)  # Initial coefficient for token 10849

# Function to calculate the steering vector
def calculate_steering_vector():
    """Calculates the steering vector based on the stored coefficients."""
    steering_vector = t.zeros_like(sae.W_dec[0])
    for token_id, coeff in steering_coeffs.items():
        steering_vector += coeff * sae.W_dec[token_id]
    return steering_vector

# --- Hook Function with Steering Vector Control ---
def steering_hook(resid_pre, hook):
    if resid_pre.shape[1] == 1:
        return

    position = sae_out.shape[1]
    if steering_on:
        # Calculate the steering vector dynamically
        steering_vector = calculate_steering_vector()
        resid_pre[:, :position - 1, :] += coeff * steering_vector


# --- Generation Function ---
def hooked_generate(prompt_batch, fwd_hooks=[], seed=None, **kwargs):
    if seed is not None:
        t.manual_seed(seed)

    with model.hooks(fwd_hooks=fwd_hooks):
        tokenized = model.to_tokens(prompt_batch)
        result = model.generate(
            stop_at_eos=False,  # avoids a bug on MPS
            input=tokenized,
            max_new_tokens=50,
            do_sample=True,
            **kwargs
        )
    return result


def run_generate(example_prompt):
    model.reset_hooks()
    editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook)]
    res = hooked_generate([example_prompt] * 3, editing_hooks, seed=None, **sampling_kwargs)

    # Print results, removing the ugly beginning of sequence token
    res_str = model.to_string(res[:, 1:])
    print(("\n\n" + "-" * 80 + "\n\n").join(res_str))

# --- Example Usage ---
example_prompt = "Who are you?"
coeff = 1
sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)

# Steering off
print("Generation without steering\n=====================================")
steering_on = False
run_generate(example_prompt)

print("\n=====================================\n\nGeneration with steering\n=====================================")
# Steering on
steering_on = True
run_generate(example_prompt)

print("\n=====================================")

# Add another coefficient for a different token
set_steering_coeff(12345, -75)
print("Generation with updated steering vector\n=====================================")
steering_on = True
run_generate(example_prompt)

print("\n=====================================")
