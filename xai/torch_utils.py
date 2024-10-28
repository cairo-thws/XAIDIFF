import torch
import numpy as np

@torch.no_grad()
def lp_dist_pd(a: torch.Tensor, b: torch.Tensor, p=1):
    # Check that both tensors are normalised to [-1,1], then rescale to [0,255]
    assert a.min() < 0 and a.min() >= -1 and a.max() <= 1
    assert b.min() < 0 and b.min() >= -1 and b.max() <= 1
    a = ((a + 1) / 2) * 255
    b = ((b + 1) / 2) * 255
    return ((a - b).norm(p=p, dim=list(range(1, len(a.shape)))) / a.numel()).detach().cpu()


# From openai guided-diffusion: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/losses.py#L50
def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

# based on OpenAI guided-diffusion/iDDPM https://github.com/openai/improved-diffusion
def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = (-log_scales).exp()
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = cdf_plus.clamp(min=1e-12).log()
    log_one_minus_cdf_min = (1.0 - cdf_min).log().clamp(min=1e-12)
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, cdf_delta.clamp(min=1e-12).log()),
    )
    assert log_probs.shape == x.shape
    return log_probs
