from functools import partial
from transformer_lens import utils

import torch as t
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate

if t.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if t.cuda.is_available() else "cpu"

# Load the model
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

# Load the sparse autoencoder
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device=device
)

# Load the dataset
dataset = load_dataset(
    path="NeelNanda/pile-10k",
    split="train",
    streaming=False,
)

token_dataset = tokenize_and_concatenate(
    dataset=dataset,
    tokenizer=model.tokenizer,
    streaming=True,
    max_length=sae.cfg.context_size,
    add_bos_token=sae.cfg.prepend_bos,
)

# Test the IOI thing
example_prompt = "When John and Mary went to the shops, John gave the bag to"
example_answer = " Mary"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

logits, cache = model.run_with_cache(example_prompt, prepend_bos=True)
tokens = model.to_tokens(example_prompt)
sae_out = sae(cache[sae.cfg.hook_name])

# Define hooks
def reconstr_hook(activations, hook, sae_out, column_to_zero=5):
    steered_activations = activations.clone()
    steered_activations[:, :, column_to_zero] = 0
    return steered_activations

def zero_abl_hook(mlp_out, hook):
    return t.zeros_like(mlp_out)

hook_name = sae.cfg.hook_name

# Run with hooks
print("Orig", model(tokens, return_type="loss").item())
print(
    "reconstr",
    model.run_with_hooks(
        tokens,
        fwd_hooks=[
            (
                hook_name,
                partial(reconstr_hook, sae_out=sae_out),
            )
        ],
        return_type="loss",
    ).item(),
)
print(
    "Zero",
    model.run_with_hooks(
        tokens,
        return_type="loss",
        fwd_hooks=[(hook_name, zero_abl_hook)],
    ).item(),
)

# Run with hooks and test prompt
with model.hooks(
    fwd_hooks=[
        (
            hook_name,
            partial(reconstr_hook, sae_out=sae_out),
        )
    ]
):
    utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)

# Example of using the hook with different column to zero
with model.hooks(
    fwd_hooks=[
        (
            hook_name,
            partial(reconstr_hook, sae_out=sae_out, column_to_zero=10),
        )
    ]
):
    print("Loss with column 10 zeroed:", model(tokens, return_type="loss").item())
