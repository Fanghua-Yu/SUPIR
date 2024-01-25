import torch

from ...util import default, instantiate_from_config


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        return log_sigma.exp()


class DiscreteSampling:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True, idx_range=None):
        self.num_idx = num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        self.idx_range = idx_range

    def idx_to_sigma(self, idx):
        # print(self.sigmas[idx])
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        if self.idx_range is None:
            idx = default(
                rand,
                torch.randint(0, self.num_idx, (n_samples,)),
            )
        else:
            idx = default(
                rand,
                torch.randint(self.idx_range[0], self.idx_range[1], (n_samples,)),
            )
        return self.idx_to_sigma(idx)

