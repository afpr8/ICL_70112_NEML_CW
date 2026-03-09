# Code to generate synthetic data for evaluating LAND

import math

import torch


def sample_non_linear_data(
    n_samples: int = 300,
    n_components: int = 20,
    x_rad: float = 0.75,
    y_rad: float = 1.5,
    std: float = 0.15,
    device="cpu",
    return_labels: bool = False,
    seed: int | None = None,
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
    """
    Generate samples from a mixture of gaussian models centered along
    a half-ellipsoidal curve
    Params:
        n_samples (optional): The number of samples to return
        n_components (optional): The number of Gaussians to sample from
        x_rad (optional): The x-radius of the ellipsoidal curve
        y_rad (optional): The y-radius of the ellipsoidal curve
        std (optional): The std for the Gaussians
        device (optional): The device used for computation
        return_labels (optional): Return component ids used to generate samples
        seed (optional): RNG seed for reproducibility
    Returns:
        samples (n_samples, 2): torch.tensor of samples x,y coords
        means (n_components, 2): torch.tensor of Gaussian mean x,y coords
        component_ids (n_samples,): torch.tensor of generating component indices
            only returned when return_labels=True
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    # Evenly divide the half-ellipsoidal curve for the Gaussian components
    t = torch.linspace(0, math.pi, n_components, device=device)

    # Define the Gaussian means in x,y coords
    means = torch.stack([
        x_rad * torch.cos(t),
        y_rad * torch.sin(t)
    ], dim=1)

    # Create samples
    component_ids = torch.randint(
        0,
        n_components,
        (n_samples,),
        device=device,
        generator=generator,
    )
    noise = std * torch.randn(n_samples, 2, device=device, generator=generator)
    samples = means[component_ids] + noise

    if return_labels:
        return samples, means, component_ids

    return samples, means
