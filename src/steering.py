import torch
from transformer_lens import HookedTransformer

def generate_with_steering(
    prompt,
    steering_vector,
    alpha,
    layer,
    model_name="gpt2",
    max_new_tokens=40,
    device="cuda"
):
    model = HookedTransformer.from_pretrained(model_name, device=device)
    steering_vector = steering_vector.to(device)

    def hook(resid, hook):
        resid[:, -1, :] += alpha * steering_vector
        return resid

    with model.hooks(
        fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook)]
    ):
        output = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )

    return output
