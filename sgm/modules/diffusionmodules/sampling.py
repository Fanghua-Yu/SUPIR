"""
    Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""


from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (
    get_ancestral_step,
    linear_multistep_coeff,
    to_d,
    to_neg_log_sigma,
    to_sigma,
)
from ...util import append_dims, default, instantiate_from_config
from k_diffusion.sampling import get_sigmas_karras, BrownianTreeNoiseSampler

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        # print('denoised', denoised.mean(axis=[0, 2, 3]))
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(
            euler_step, x, d, dt, next_sigma, denoiser, cond, uc
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(
                *self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs
            )
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [
                linear_multistep_coeff(cur_order, sigmas_cpu, i, j)
                for j in range(cur_order)
            ]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        # print("euler_step: ", euler_step.mean(axis=[0, 2, 3]))
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(
        self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc
    ):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step
            )
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [
                append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)
            ]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x


class SubstepSampler(EulerAncestralSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, restore_cfg=4.0,
            restore_cfg_s_tmin=0.05, eta=1., n_sample_steps=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(1000, device=self.device)
        sigmas = sigmas[
            self.steps_subset[: self.num_steps] + self.steps_subset[-1:]
        ]
        print(sigmas)
        # uc = cond
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc, control_scale=1.0):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), control_scale)
        denoised = self.guider(denoised, sigma)
        return denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, control_scale=1.0, *args, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                control_scale=control_scale,
            )

        return x

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, control_scale=1.0):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc, control_scale=control_scale)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class RestoreDPMPP2MSampler(DPMPP2MSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, restore_cfg=4.0,
            restore_cfg_s_tmin=0.05, eta=1., *args, **kwargs):
        self.s_noise = s_noise
        self.eta = eta
        super().__init__(*args, **kwargs)

    def denoise(self, x, denoiser, sigma, cond, uc, control_scale=1.0):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), control_scale)
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_mult(self, h, r, t, t_next, previous_sigma):
        eta_h = self.eta * h
        mult1 = to_sigma(t_next) / to_sigma(t) * (-eta_h).exp()
        mult2 = (-h -eta_h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2


    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
        eps_noise=None,
        control_scale=1.0,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc, control_scale=control_scale)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        eta_h = self.eta * h
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, t, t_next, previous_sigma)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(
                append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard
            )
            if self.eta:
                x = x + eps_noise * next_sigma * (-2 * eta_h).expm1().neg().sqrt() * self.s_noise

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, control_scale=1.0, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
        sigmas_min, sigmas_max = sigmas[-2].cpu(), sigmas[0].cpu()
        sigmas_new = get_sigmas_karras(self.num_steps, sigmas_min, sigmas_max, device=x.device)
        sigmas = sigmas_new

        noise_sampler = BrownianTreeNoiseSampler(x, sigmas_min, sigmas_max)

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            if i > 0 and torch.sum(s_in * sigmas[i + 1]) > 1e-14:
                eps_noise = noise_sampler(s_in * sigmas[i], s_in * sigmas[i + 1])
            else:
                eps_noise = None
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
                eps_noise=eps_noise,
                control_scale=control_scale,
            )

        return x


def to_d_center(denoised, x_center, x):
    b = denoised.shape[0]
    v_center = (denoised - x_center).view(b, -1)
    v_denoise = (x - denoised).view(b, -1)
    d_center = v_center - v_denoise * (v_center * v_denoise).sum(dim=1).view(b, 1) / \
                (v_denoise * v_denoise).sum(dim=1).view(b, 1)
    d_center = d_center / d_center.view(x.shape[0], -1).norm(dim=1).view(-1, 1)
    return d_center.view(denoised.shape)


class RestoreEDMSampler(SingleStepDiffusionSampler):
    def __init__(
        self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, restore_cfg=4.0,
            restore_cfg_s_tmin=0.05, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise
        self.restore_cfg = restore_cfg
        self.restore_cfg_s_tmin = restore_cfg_s_tmin
        self.sigma_max = 14.6146

    def denoise(self, x, denoiser, sigma, cond, uc, control_scale=1.0):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), control_scale)
        denoised = self.guider(denoised, sigma)
        return denoised

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0, x_center=None, eps_noise=None,
                     control_scale=1.0, use_linear_control_scale=False, control_scale_start=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            if eps_noise is not None:
                eps = eps_noise * self.s_noise
            else:
                eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        if use_linear_control_scale:
            control_scale = (sigma[0].item() / self.sigma_max) * (control_scale_start - control_scale) + control_scale

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc, control_scale=control_scale)

        if (next_sigma[0] > self.restore_cfg_s_tmin) and (self.restore_cfg > 0):
            d_center = (denoised - x_center)
            denoised = denoised - d_center * ((sigma.view(-1, 1, 1, 1) / self.sigma_max) ** self.restore_cfg)

        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)
        x = self.euler_step(x, d, dt)
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, x_center=None, control_scale=1.0,
                 use_linear_control_scale=False, control_scale_start=0.0):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for _idx, i in enumerate(self.get_sigma_gen(num_sigmas)):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
                x_center,
                control_scale=control_scale,
                use_linear_control_scale=use_linear_control_scale,
                control_scale_start=control_scale_start,
            )
        return x


class TiledRestoreEDMSampler(RestoreEDMSampler):
    def __init__(self, tile_size=128, tile_stride=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.tile_weights = gaussian_weights(self.tile_size, self.tile_size, 1)

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, x_center=None, control_scale=1.0,
                 use_linear_control_scale=False, control_scale_start=0.0):
        use_local_prompt = isinstance(cond, list)
        b, _, h, w = x.shape
        latent_tiles_iterator = _sliding_windows(h, w, self.tile_size, self.tile_stride)
        tile_weights = self.tile_weights.repeat(b, 1, 1, 1)
        if not use_local_prompt:
            LQ_latent = cond['control']
        else:
            assert len(cond) == len(latent_tiles_iterator), "Number of local prompts should be equal to number of tiles"
            LQ_latent = cond[0]['control']
        clean_LQ_latent = x_center
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for _idx, i in enumerate(self.get_sigma_gen(num_sigmas)):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1)
                if self.s_tmin <= sigmas[i] <= self.s_tmax
                else 0.0
            )
            x_next = torch.zeros_like(x)
            count = torch.zeros_like(x)
            eps_noise = torch.randn_like(x)
            for j, (hi, hi_end, wi, wi_end) in enumerate(latent_tiles_iterator):
                x_tile = x[:, :, hi:hi_end, wi:wi_end]
                _eps_noise = eps_noise[:, :, hi:hi_end, wi:wi_end]
                x_center_tile = clean_LQ_latent[:, :, hi:hi_end, wi:wi_end]
                if use_local_prompt:
                    _cond = cond[j]
                else:
                    _cond = cond
                _cond['control'] = LQ_latent[:, :, hi:hi_end, wi:wi_end]
                uc['control'] = LQ_latent[:, :, hi:hi_end, wi:wi_end]
                _x = self.sampler_step(
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x_tile,
                    _cond,
                    uc,
                    gamma,
                    x_center_tile,
                    eps_noise=_eps_noise,
                    control_scale=control_scale,
                    use_linear_control_scale=use_linear_control_scale,
                    control_scale_start=control_scale_start,
                )
                x_next[:, :, hi:hi_end, wi:wi_end] += _x * tile_weights
                count[:, :, hi:hi_end, wi:wi_end] += tile_weights
            x_next /= count
            x = x_next
        return x


class TiledRestoreDPMPP2MSampler(RestoreDPMPP2MSampler):
    def __init__(self, tile_size=128, tile_stride=64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_size = tile_size
        self.tile_stride = tile_stride
        self.tile_weights = gaussian_weights(self.tile_size, self.tile_size, 1)

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, control_scale=1.0, **kwargs):
        use_local_prompt = isinstance(cond, list)
        b, _, h, w = x.shape
        latent_tiles_iterator = _sliding_windows(h, w, self.tile_size, self.tile_stride)
        tile_weights = self.tile_weights.repeat(b, 1, 1, 1)
        if not use_local_prompt:
            LQ_latent = cond['control']
        else:
            assert len(cond) == len(latent_tiles_iterator), "Number of local prompts should be equal to number of tiles"
            LQ_latent = cond[0]['control']
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )
        sigmas_min, sigmas_max = sigmas[-2].cpu(), sigmas[0].cpu()
        sigmas_new = get_sigmas_karras(self.num_steps, sigmas_min, sigmas_max, device=x.device)
        sigmas = sigmas_new

        noise_sampler = BrownianTreeNoiseSampler(x, sigmas_min, sigmas_max)

        old_denoised = None
        for _idx, i in enumerate(self.get_sigma_gen(num_sigmas)):
            if i > 0 and torch.sum(s_in * sigmas[i + 1]) > 1e-14:
                eps_noise = noise_sampler(s_in * sigmas[i], s_in * sigmas[i + 1])
            else:
                eps_noise = torch.zeros_like(x)
            x_next = torch.zeros_like(x)
            old_denoised_next = torch.zeros_like(x)
            count = torch.zeros_like(x)
            for j, (hi, hi_end, wi, wi_end) in enumerate(latent_tiles_iterator):
                x_tile = x[:, :, hi:hi_end, wi:wi_end]
                _eps_noise = eps_noise[:, :, hi:hi_end, wi:wi_end]
                if old_denoised is not None:
                    old_denoised_tile = old_denoised[:, :, hi:hi_end, wi:wi_end]
                else:
                    old_denoised_tile = None
                if use_local_prompt:
                    _cond = cond[j]
                else:
                    _cond = cond
                _cond['control'] = LQ_latent[:, :, hi:hi_end, wi:wi_end]
                uc['control'] = LQ_latent[:, :, hi:hi_end, wi:wi_end]
                _x, _old_denoised = self.sampler_step(
                    old_denoised_tile,
                    None if i == 0 else s_in * sigmas[i - 1],
                    s_in * sigmas[i],
                    s_in * sigmas[i + 1],
                    denoiser,
                    x_tile,
                    _cond,
                    uc=uc,
                    eps_noise=_eps_noise,
                    control_scale=control_scale,
                )
                x_next[:, :, hi:hi_end, wi:wi_end] += _x * tile_weights
                old_denoised_next[:, :, hi:hi_end, wi:wi_end] += _old_denoised * tile_weights
                count[:, :, hi:hi_end, wi:wi_end] += tile_weights
            old_denoised_next /= count
            x_next /= count
            x = x_next
            old_denoised = old_denoised_next
        return x


def gaussian_weights(tile_width, tile_height, nbatches):
    """Generates a gaussian mask of weights for tile contributions"""
    from numpy import pi, exp, sqrt
    import numpy as np

    latent_width = tile_width
    latent_height = tile_height

    var = 0.01
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var)
               for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var)
               for y in range(latent_height)]

    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device='cuda'), (nbatches, 4, 1, 1))


def _sliding_windows(h: int, w: int, tile_size: int, tile_stride: int):
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)

    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)

    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords
