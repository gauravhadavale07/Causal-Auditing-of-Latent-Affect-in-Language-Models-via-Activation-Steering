import torch

def mean_difference_vector(a, b):
    return a.mean(dim=0) - b.mean(dim=0)


if __name__ == "__main__":
    anger = torch.load("anger_activations.pt")
    neutral = torch.load("neutral_activations.pt")

    anger_vector = mean_difference_vector(anger, neutral)
    anger_vector = anger_vector / anger_vector.norm()

    control_vector = torch.randn_like(anger_vector)
    control_vector = control_vector / control_vector.norm()

    torch.save(anger_vector, "anger_vector.pt")
    torch.save(control_vector, "control_vector.pt")
