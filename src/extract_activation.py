import torch
from transformer_lens import HookedTransformer

def extract_last_token_residuals(
    prompts,
    layer,
    model_name="gpt2",
    device="cuda"
):
    model = HookedTransformer.from_pretrained(model_name, device=device)
    acts = []

    for prompt in prompts:
        _, cache = model.run_with_cache(prompt)
        resid = cache[f"blocks.{layer}.hook_resid_post"]
        acts.append(resid[:, -1, :].detach().cpu())

    return torch.cat(acts, dim=0)


if __name__ == "__main__":
    prompts = open("data/prompts.txt").read().splitlines()
    acts = extract_last_token_residuals(prompts, layer=8)
    torch.save(acts, "anger_activations.pt")
