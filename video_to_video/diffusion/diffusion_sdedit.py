import random

import torch

from video_to_video.utils.logger import get_logger
from video_to_video.utils.util import blend_time
from .schedules_sdedit import karras_schedule
from .solvers_sdedit import sample_dpmpp_2m_sde, sample_heun

logger = get_logger()

__all__ = ["GaussianDiffusion"]


def _i(tensor, t, x):
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    return tensor[t.to(tensor.device)].view(shape).to(x.device)


class GaussianDiffusion(object):

    def __init__(self, sigmas):
        self.sigmas = sigmas
        self.alphas = torch.sqrt(1 - sigmas**2)
        self.num_timesteps = len(sigmas)

    def diffuse(self, x0, t, noise=None):
        noise = torch.randn_like(x0) if noise is None else noise
        xt = _i(self.alphas, t, x0) * x0 + _i(self.sigmas, t, x0) * noise
        return xt

    def denoise(
        self, xt, t, s, model, model_kwargs={}, guide_scale=None, guide_rescale=None, clamp=None, percentile=None
    ):
        s = t - 1 if s is None else s

        # hyperparams
        sigmas = _i(self.sigmas, t, xt)
        alphas = _i(self.alphas, t, xt)
        alphas_s = _i(self.alphas, s.clamp(0), xt)
        alphas_s[s < 0] = 1.0
        sigmas_s = torch.sqrt(1 - alphas_s**2)

        # precompute variables
        betas = 1 - (alphas / alphas_s) ** 2
        coef1 = betas * alphas_s / sigmas**2
        coef2 = (alphas * sigmas_s**2) / (alphas_s * sigmas**2)
        var = betas * (sigmas_s / sigmas) ** 2
        log_var = torch.log(var).clamp_(-20, 20)

        # prediction
        if guide_scale is None:
            assert isinstance(model_kwargs, dict)
            out = model(xt, t=t, **model_kwargs)
        else:
            # classifier-free guidance
            assert isinstance(model_kwargs, list)
            if len(model_kwargs) > 2:
                y_out = model(
                    xt,
                    t=t,
                    **model_kwargs[0],
                    **model_kwargs[2],
                    **model_kwargs[3],
                    **model_kwargs[4],
                    **model_kwargs[5],
                )
            else:
                y_out = model(xt, t=t, **model_kwargs[0])
            if guide_scale == 1.0:
                out = y_out
            else:
                if len(model_kwargs) > 2:
                    u_out = model(
                        xt,
                        t=t,
                        **model_kwargs[1],
                        **model_kwargs[2],
                        **model_kwargs[3],
                        **model_kwargs[4],
                        **model_kwargs[5],
                    )
                else:
                    u_out = model(xt, t=t, **model_kwargs[1])
                out = u_out + guide_scale * (y_out - u_out)

                if guide_rescale is not None:
                    assert guide_rescale >= 0 and guide_rescale <= 1
                    ratio = (y_out.flatten(1).std(dim=1) / (out.flatten(1).std(dim=1) + 1e-12)).view(  # noqa
                        (-1,) + (1,) * (y_out.ndim - 1)
                    )
                    out *= guide_rescale * ratio + (1 - guide_rescale) * 1.0

        x0 = alphas * xt - sigmas * out

        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1
            s = torch.quantile(x0.flatten(1).abs(), percentile, dim=1)
            s = s.clamp_(1.0).view((-1,) + (1,) * (xt.ndim - 1))
            x0 = torch.min(s, torch.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)

        # recompute eps using the restricted x0
        eps = (xt - alphas * x0) / sigmas

        # compute mu (mean of posterior distribution) using the restricted x0
        mu = coef1 * x0 + coef2 * xt
        return mu, var, log_var, x0, eps

    @torch.no_grad()
    def sample(
        self,
        noise,
        model,
        model_kwargs={},
        condition_fn=None,
        guide_scale=None,
        guide_rescale=None,
        clamp=None,
        percentile=None,
        solver="euler_a",
        solver_mode="fast",
        steps=20,
        t_max=None,
        t_min=None,
        discretization=None,
        discard_penultimate_step=None,
        return_intermediate=None,
        show_progress=False,
        seed=-1,
        chunk_inds=None,
        rotation_decoding=False,
        **kwargs,
    ):
        # sanity check
        assert isinstance(steps, (int, torch.LongTensor))
        assert t_max is None or (t_max > 0 and t_max <= self.num_timesteps - 1)
        assert t_min is None or (t_min >= 0 and t_min < self.num_timesteps - 1)
        assert discretization in (None, "leading", "linspace", "trailing")
        assert discard_penultimate_step in (None, True, False)
        assert return_intermediate in (None, "x0", "xt")

        # function of diffusion solver
        solver_fn = {"heun": sample_heun, "dpmpp_2m_sde": sample_dpmpp_2m_sde}[solver]

        # options
        schedule = "karras" if "karras" in solver else None
        discretization = discretization or "linspace"
        seed = seed if seed >= 0 else random.randint(0, 2**31)
        if isinstance(steps, torch.LongTensor):
            discard_penultimate_step = False
        if discard_penultimate_step is None:
            discard_penultimate_step = (
                True
                if solver
                in (
                    "dpm2",
                    "dpm2_ancestral",
                    "dpmpp_2m_sde",
                    "dpm2_karras",
                    "dpm2_ancestral_karras",
                    "dpmpp_2m_sde_karras",
                )
                else False
            )

        # function for denoising xt to get x0
        intermediates = []

        def model_fn(xt, sigma):
            # denoising
            t = self._sigma_to_t(sigma).repeat(len(xt)).round().long()
            x0 = self.denoise(xt, t, None, model, model_kwargs, guide_scale, guide_rescale, clamp, percentile)[-2]

            if rotation_decoding:
                model_kwargs[2]['hint'] = model_kwargs[2]['hint'].roll(model_kwargs[2]['hint'].shape[-1] // 4, dims=-1)

            # collect intermediate outputs
            if return_intermediate == "xt":
                intermediates.append(xt)
            elif return_intermediate == "x0":
                intermediates.append(x0)
            return x0

        mask_cond = model_kwargs[3]["mask_cond"]

        def model_chunk_fn(xt, sigma):
            # denoising
            t = self._sigma_to_t(sigma).repeat(len(xt)).round().long()
            overlap_time = chunk_inds[0][1] - chunk_inds[1][0] if len(chunk_inds) > 1 else 0

            time = []
            for i in range(len(chunk_inds)):
                ind_start, ind_end = chunk_inds[i]
                xt_chunk = xt[:, :, ind_start:ind_end].clone()
                model_kwargs[3]["mask_cond"] = mask_cond[:, ind_start:ind_end].clone()
                x0_chunk = self.denoise(
                    xt_chunk, t, None, model, model_kwargs, guide_scale, guide_rescale, clamp, percentile
                )[-2]
                time.append(x0_chunk)
            blended_time = []
            for k, chunk in enumerate(time):
                chunk_size = chunk.size(2)
                if k > 0:
                    chunk = blend_time(time[k - 1], chunk, overlap_time)
                if k != len(time) - 1:
                    blended_time.append(chunk[:, :, : chunk_size - overlap_time])
                else:
                    blended_time.append(chunk)
            x0 = torch.concat(blended_time, dim=2)
            torch.cuda.empty_cache()

            if rotation_decoding:
                model_kwargs[2]['hint'] = model_kwargs[2]['hint'].roll(model_kwargs[2]['hint'].shape[-1] // 4, dims=-1)

            return x0

        # get timesteps
        if isinstance(steps, int):
            steps += 1 if discard_penultimate_step else 0
            t_max = self.num_timesteps - 1 if t_max is None else t_max
            t_min = 0 if t_min is None else t_min

            # discretize timesteps
            if discretization == "leading":
                steps = torch.arange(t_min, t_max + 1, (t_max - t_min + 1) / steps).flip(0)
            elif discretization == "linspace":
                steps = torch.linspace(t_max, t_min, steps)
            elif discretization == "trailing":
                steps = torch.arange(t_max, t_min - 1, -((t_max - t_min + 1) / steps))
                if solver_mode == "fast":
                    t_mid = 500
                    steps1 = torch.arange(t_max, t_mid - 1, -((t_max - t_mid + 1) / 4))
                    steps2 = torch.arange(t_mid, t_min - 1, -((t_mid - t_min + 1) / 11))
                    steps = torch.concat([steps1, steps2])
            else:
                raise NotImplementedError(f"{discretization} discretization not implemented")
            steps = steps.clamp_(t_min, t_max)
        steps = torch.as_tensor(steps, dtype=torch.float32, device=noise.device)

        # get sigmas
        sigmas = self._t_to_sigma(steps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if schedule == "karras":
            if sigmas[0] == float("inf"):
                sigmas = karras_schedule(
                    n=len(steps) - 1,
                    sigma_min=sigmas[sigmas > 0].min().item(),
                    sigma_max=sigmas[sigmas < float("inf")].max().item(),
                    rho=7.0,
                ).to(sigmas)
                sigmas = torch.cat([sigmas.new_tensor([float("inf")]), sigmas, sigmas.new_zeros([1])])
            else:
                sigmas = karras_schedule(
                    n=len(steps), sigma_min=sigmas[sigmas > 0].min().item(), sigma_max=sigmas.max().item(), rho=7.0
                ).to(sigmas)
                sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        if discard_penultimate_step:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        fn = model_chunk_fn if chunk_inds is not None else model_fn
        x0 = solver_fn(noise, fn, sigmas, show_progress=show_progress, rotation_decoding=rotation_decoding, **kwargs)
        return (x0, intermediates) if return_intermediate is not None else x0

    def _sigma_to_t(self, sigma):
        if sigma == float("inf"):
            t = torch.full_like(sigma, len(self.sigmas) - 1)
        else:
            log_sigmas = torch.sqrt(self.sigmas**2 / (1 - self.sigmas**2)).log().to(sigma)  # noqa
            log_sigma = sigma.log()
            dists = log_sigma - log_sigmas[:, None]
            low_idx = dists.ge(0).cumsum(dim=0).argmax(dim=0).clamp(max=log_sigmas.shape[0] - 2)
            high_idx = low_idx + 1
            low, high = log_sigmas[low_idx], log_sigmas[high_idx]
            w = (low - log_sigma) / (low - high)
            w = w.clamp(0, 1)
            t = (1 - w) * low_idx + w * high_idx
            t = t.view(sigma.shape)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return t

    def _t_to_sigma(self, t):
        t = t.float()
        low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
        log_sigmas = torch.sqrt(self.sigmas**2 / (1 - self.sigmas**2)).log().to(t)  # noqa
        log_sigma = (1 - w) * log_sigmas[low_idx] + w * log_sigmas[high_idx]
        log_sigma[torch.isnan(log_sigma) | torch.isinf(log_sigma)] = float("inf")
        return log_sigma.exp()
