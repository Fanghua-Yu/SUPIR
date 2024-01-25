import torch
import torch.nn as nn
from packaging import version
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.cache_size_limit = 512

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )


class OpenAIHalfWrapper(IdentityWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_model = self.diffusion_model.half()

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        _context = c.get("crossattn", None)
        _y = c.get("vector", None)
        if _context is not None:
            _context = _context.half()
        if _y is not None:
            _y = _y.half()
        x = x.half()
        t = t.half()

        out = self.diffusion_model(
            x,
            timesteps=t,
            context=_context,
            y=_y,
            **kwargs,
        )
        return out.float()


class ControlWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False, dtype=torch.float32):
        super().__init__()
        self.compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = self.compile(diffusion_model)
        self.control_model = None
        self.dtype = dtype

    def load_control_model(self, control_model):
        self.control_model = self.compile(control_model)

    def forward(
            self, x: torch.Tensor, t: torch.Tensor, c: dict, control_scale=1, **kwargs
    ) -> torch.Tensor:
        with torch.autocast("cuda", dtype=self.dtype):
            control = self.control_model(x=c.get("control", None), timesteps=t, xt=x,
                                         control_vector=c.get("control_vector", None),
                                         mask_x=c.get("mask_x", None),
                                         context=c.get("crossattn", None),
                                         y=c.get("vector", None))
            out = self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                control=control,
                control_scale=control_scale,
                **kwargs,
            )
        return out.float()

