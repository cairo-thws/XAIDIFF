"""
bpd evaluation based on OpenAI iDDPM https://github.com/openai/improved-diffusion
"""

from xai.torch_utils import discretized_gaussian_log_likelihood

# OpenAI uses th instead of torch alias
import torch as th
import numpy as np


def calc_bpd_loop(pipeline, x_start, clip_denoised=True, model_kwargs=None):
    """
    Compute the entire variational lower-bound, measured in bits-per-dim,
    as well as other related quantities.

    :param pipeline: the model to evaluate loss on.
    :param x_start: the [N x C x ...] tensor of inputs.
    :param clip_denoised: if True, clip denoised samples.

    :return: a dict containing the following keys:
                - total_bpd: the total variational lower-bound, per batch element.
                - prior_bpd: the prior term in the lower-bound.
                - vb: an [N x T] tensor of terms in the lower-bound.
                - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                - mse: an [N x T] tensor of epsilon MSEs for each timestep.
    """
    device = x_start.device
    batch_size = x_start.shape[0]

    vb = []
    xstart_mse = []
    mse = []

    # extract and precalculate constants
    alphas = pipeline.scheduler.alphas
    betas = pipeline.scheduler.betas
    alphas_cumprod = pipeline.scheduler.alphas_cumprod
    alphas_cumprod_prev = th.cat((th.ones(1), alphas_cumprod[:-1]))
    # We do not need this
    # alphas_cumprod_next = th.cat((alphas_cumprod[1:], th.zeros_like(alphas_cumprod[0])))

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = th.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = th.sqrt(1.0 - alphas_cumprod)
    log_one_minus_alphas_cumprod = th.log(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = th.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / alphas_cumprod - 1)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    # log calculation clipped because the posterior variance is 0 at the
    # beginning of the diffusion chain.
    posterior_log_variance_clipped = th.cat((posterior_variance[[1]], posterior_variance[1:])).log()

    posterior_mean_coef1 = betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod)
    posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod)

    def _q_sample(x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _expand_to_data(sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _expand_to_data(sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
    #     return (
    #         _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
    #         - pred_xstart
    #     ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior_mean_variance(x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _expand_to_data(posterior_mean_coef1, t, x_t.shape) * x_start
            + _expand_to_data(posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_var = _expand_to_data(posterior_variance, t, x_t.shape)
        posterior_log_var_clipped = _expand_to_data(posterior_log_variance_clipped, t, x_t.shape)
        assert (
            posterior_mean.shape[0] == posterior_var.shape[0] == posterior_log_var_clipped.shape[0] == x_start.shape[0]
        )
        return posterior_mean, posterior_var, posterior_log_var_clipped

    def p_mean_variance(pipeline, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param pipeline: pipeline with the model, which takes a signal and a batch of timesteps
                        as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                    - 'mean': the model mean output.
                    - 'variance': the model variance output.
                    - 'log_variance': the log of 'variance'.
                    - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = pipeline.unet(x.to(pipeline.device), t.to(pipeline.device))["sample"].cpu()

        model_variance = _expand_to_data(posterior_variance, t, x.shape)
        model_log_variance = _expand_to_data(posterior_log_variance_clipped, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(_predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
        model_mean, _, _ = q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "pred_eps": model_output,
        }

    def _predict_xstart_from_eps(x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _expand_to_data(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _expand_to_data(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_mean_variance(x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _expand_to_data(sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _expand_to_data(1.0 - alphas_cumprod, t, x_start.shape)
        log_variance = _expand_to_data(log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def _prior_bpd(x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def _vb_terms_bpd(model, x_start, x_t, t, clip_denoised=True, model_kwargs=None):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                    - 'output': a shape [N] tensor of NLLs or KLs.
                    - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = p_mean_variance(model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {
            "output": output,
            "pred_xstart": out["pred_xstart"],
            "pred_eps": out["pred_eps"],
        }

    num_timesteps = pipeline.scheduler.config.num_train_timesteps
    # num_timesteps = 2
    for t in list(range(num_timesteps))[::-1]:
        t_batch = th.tensor([t] * batch_size, device=device)
        noise = th.randn_like(x_start)
        x_t = _q_sample(x_start=x_start, t=t_batch, noise=noise)
        # Calculate VLB term at the current timestep
        with th.no_grad():
            out = _vb_terms_bpd(
                pipeline,
                x_start=x_start,
                x_t=x_t,
                t=t_batch,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
            )
        vb.append(out["output"])
        xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
        eps = out["pred_eps"]
        ## this was in OpenAI implementation but makes litle sense cause we have the eps alread
        # eps = _predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
        mse.append(mean_flat((eps - noise) ** 2))

    vb = th.stack(vb, dim=1)
    xstart_mse = th.stack(xstart_mse, dim=1)
    mse = th.stack(mse, dim=1)

    prior_bpd = _prior_bpd(x_start)
    total_bpd = vb.sum(dim=1) + prior_bpd

    # print(th.std_mean(total_bpd))
    return {
        "total_bpd": total_bpd,
        "prior_bpd": prior_bpd,
        "vb": vb,
        "xstart_mse": xstart_mse,
        "mse": mse,
    }


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor) for x in (logvar1, logvar2)]

    return 0.5 * (-1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * th.exp(-logvar2))


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def _expand_to_data(res, timesteps, broadcast_shape):
    """
    Expand tensor to match data tensors for broadcasting

    :param res: the 1-D tensor
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = res[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
