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

steering_vector = 150 * sae.W_dec[10849] # currently set manually. this is the thing that we want to learn/optimize for/etc
# in general, the optimal solution is going to look something like
# steering_vector = coeffs[0] * sae.W_dec[0] + coeffs[1] * sae.W_dec[1] + ... + coeffs[n] * sae.W_dec[n]
# and finding these coefficients is the relevant search problem

example_prompt = "Who are you?"
coeff = 1
sampling_kwargs = dict(temperature=1.0, top_p=0.1, freq_penalty=1.0)


def steering_hook(resid_pre, hook):
    if resid_pre.shape[1] == 1:
        return

    position = sae_out.shape[1]
    if steering_on:
        # using our steering vector and applying the coefficient
        resid_pre[:, :position - 1, :] += coeff * steering_vector


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
            **kwargs)
    return result


def run_generate(example_prompt):
    model.reset_hooks()
    editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook)]
    res = hooked_generate([example_prompt] * 3, editing_hooks, seed=None, **sampling_kwargs)

    # Print results, removing the ugly beginning of sequence token
    res_str = model.to_string(res[:, 1:])
    print(("\n\n" + "-" * 80 + "\n\n").join(res_str))


# steering_on = False
# print("Generation without steering\n=====================================")
# run_generate(example_prompt)
print("\n=====================================\n\nGeneration with steering\n=====================================")
steering_on = True
run_generate(example_prompt)
print("\n=====================================")
