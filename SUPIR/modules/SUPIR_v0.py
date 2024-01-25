# from einops._torch_specific import allow_ops_in_compiled_graph
# allow_ops_in_compiled_graph()
import einops
import torch
import torch as th
import torch.nn as nn
from einops import rearrange, repeat

from sgm.modules.diffusionmodules.util import (
    avg_pool_nd,
    checkpoint,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)

from sgm.modules.diffusionmodules.openaimodel import Downsample, Upsample, UNetModel, Timestep, \
    TimestepEmbedSequential, ResBlock, AttentionBlock, TimestepBlock
from sgm.modules.attention import SpatialTransformer, MemoryEfficientCrossAttention, CrossAttention
from sgm.util import default, log_txt_as_img, exists, instantiate_from_config
import re
import torch
from functools import partial


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


# dummy replace
def convert_module_to_f16(x):
    pass


def convert_module_to_f32(x):
    pass


class ZeroConv(nn.Module):
    def __init__(self, label_nc, norm_nc, mask=False):
        super().__init__()
        self.zero_conv = zero_module(conv_nd(2, label_nc, norm_nc, 1, 1, 0))
        self.mask = mask

    def forward(self, c, h, h_ori=None):
        # with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32):
        if not self.mask:
            h = h + self.zero_conv(c)
        else:
            h = h + self.zero_conv(c) * torch.zeros_like(h)
        if h_ori is not None:
            h = th.cat([h_ori, h], dim=1)
        return h


class ZeroSFT(nn.Module):
    def __init__(self, label_nc, norm_nc, concat_channels=0, norm=True, mask=False):
        super().__init__()

        # param_free_norm_type = str(parsed.group(1))
        ks = 3
        pw = ks // 2

        self.norm = norm
        if self.norm:
            self.param_free_norm = normalization(norm_nc + concat_channels)
        else:
            self.param_free_norm = nn.Identity()

        nhidden = 128

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.SiLU()
        )
        self.zero_mul = zero_module(nn.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw))
        self.zero_add = zero_module(nn.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw))
        # self.zero_mul = nn.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw)
        # self.zero_add = nn.Conv2d(nhidden, norm_nc + concat_channels, kernel_size=ks, padding=pw)

        self.zero_conv = zero_module(conv_nd(2, label_nc, norm_nc, 1, 1, 0))
        self.pre_concat = bool(concat_channels != 0)
        self.mask = mask

    def forward(self, c, h, h_ori=None, control_scale=1):
        assert self.mask is False
        if h_ori is not None and self.pre_concat:
            h_raw = th.cat([h_ori, h], dim=1)
        else:
            h_raw = h

        if self.mask:
            h = h + self.zero_conv(c) * torch.zeros_like(h)
        else:
            h = h + self.zero_conv(c)
        if h_ori is not None and self.pre_concat:
            h = th.cat([h_ori, h], dim=1)
        actv = self.mlp_shared(c)
        gamma = self.zero_mul(actv)
        beta = self.zero_add(actv)
        if self.mask:
            gamma = gamma * torch.zeros_like(gamma)
            beta = beta * torch.zeros_like(beta)
        h = self.param_free_norm(h) * (gamma + 1) + beta
        if h_ori is not None and not self.pre_concat:
            h = th.cat([h_ori, h], dim=1)
        return h * control_scale + h_raw * (1 - control_scale)


class ZeroCrossAttn(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }

    def __init__(self, context_dim, query_dim, zero_out=True, mask=False):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn = attn_cls(query_dim=query_dim, context_dim=context_dim, heads=query_dim//64, dim_head=64)
        self.norm1 = normalization(query_dim)
        self.norm2 = normalization(context_dim)

        self.mask = mask

        # if zero_out:
        #     # for p in self.attn.to_out.parameters():
        #     #     p.detach().zero_()
        #     self.attn.to_out = zero_module(self.attn.to_out)

    def forward(self, context, x, control_scale=1):
        assert self.mask is False
        x_in = x
        x = self.norm1(x)
        context = self.norm2(context)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        context = rearrange(context, 'b c h w -> b (h w) c').contiguous()
        x = self.attn(x, context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if self.mask:
            x = x * torch.zeros_like(x)
        x = x_in + x * control_scale

        return x


class GLVControl(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        spatial_transformer_attn_type="softmax",
        adm_in_channels=None,
        use_fairscale_checkpoint=False,
        offload_to_cpu=False,
        transformer_depth_middle=None,
        input_upscale=1,
    ):
        super().__init__()
        from omegaconf.listconfig import ListConfig

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        elif isinstance(transformer_depth, ListConfig):
            transformer_depth = list(transformer_depth)
        transformer_depth_middle = default(
            transformer_depth_middle, transformer_depth[-1]
        )

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        # self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        if use_fp16:
            print("WARNING: use_fp16 was dropped and has no effect anymore.")
        # self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        assert use_fairscale_checkpoint != use_checkpoint or not (
            use_checkpoint or use_fairscale_checkpoint
        )

        self.use_fairscale_checkpoint = False
        checkpoint_wrapper_fn = (
            partial(checkpoint_wrapper, offload_to_cpu=offload_to_cpu)
            if self.use_fairscale_checkpoint
            else lambda x: x
        )

        time_embed_dim = model_channels * 4
        self.time_embed = checkpoint_wrapper_fn(
            nn.Sequential(
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            )
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = checkpoint_wrapper_fn(
                    nn.Sequential(
                        Timestep(model_channels),
                        nn.Sequential(
                            linear(model_channels, time_embed_dim),
                            nn.SiLU(),
                            linear(time_embed_dim, time_embed_dim),
                        ),
                    )
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    checkpoint_wrapper_fn(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                        )
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            checkpoint_wrapper_fn(
                                AttentionBlock(
                                    ch,
                                    use_checkpoint=use_checkpoint,
                                    num_heads=num_heads,
                                    num_head_channels=dim_head,
                                    use_new_attention_order=use_new_attention_order,
                                )
                            )
                            if not use_spatial_transformer
                            else checkpoint_wrapper_fn(
                                SpatialTransformer(
                                    ch,
                                    num_heads,
                                    dim_head,
                                    depth=transformer_depth[level],
                                    context_dim=context_dim,
                                    disable_self_attn=disabled_sa,
                                    use_linear=use_linear_in_transformer,
                                    attn_type=spatial_transformer_attn_type,
                                    use_checkpoint=use_checkpoint,
                                )
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        checkpoint_wrapper_fn(
                            ResBlock(
                                ch,
                                time_embed_dim,
                                dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                            )
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            checkpoint_wrapper_fn(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            ),
            checkpoint_wrapper_fn(
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
            )
            if not use_spatial_transformer
            else checkpoint_wrapper_fn(
                SpatialTransformer(  # always uses a self-attn
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth_middle,
                    context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer,
                    attn_type=spatial_transformer_attn_type,
                    use_checkpoint=use_checkpoint,
                )
            ),
            checkpoint_wrapper_fn(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            ),
        )

        self.input_upscale = input_upscale
        self.input_hint_block = TimestepEmbedSequential(
                    zero_module(conv_nd(dims, in_channels, model_channels, 3, padding=1))
                )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps, xt, context=None, y=None, **kwargs):
        # with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32):
        #     x = x.to(torch.float32)
        #     timesteps = timesteps.to(torch.float32)
        #     xt = xt.to(torch.float32)
        #     context = context.to(torch.float32)
        #     y = y.to(torch.float32)
        # print(x.dtype)
        xt, context, y = xt.to(x.dtype), context.to(x.dtype), y.to(x.dtype)

        if self.input_upscale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.input_upscale, mode='bilinear', antialias=True)
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
        # import pdb
        # pdb.set_trace()
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == xt.shape[0]
            emb = emb + self.label_emb(y)

        guided_hint = self.input_hint_block(x, emb, context)

        # h = x.type(self.dtype)
        h = xt
        for module in self.input_blocks:
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            hs.append(h)
            # print(module)
            # print(h.shape)
        h = self.middle_block(h, emb, context)
        hs.append(h)
        return hs


class LightGLVUNet(UNetModel):
    def __init__(self, mode='', project_type='ZeroSFT', project_channel_scale=1,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mode == 'XL-base':
            cond_output_channels = [320] * 4 + [640] * 3 + [1280] * 3
            project_channels = [160] * 4 + [320] * 3 + [640] * 3
            concat_channels = [320] * 2 + [640] * 3 + [1280] * 4 + [0]
            cross_attn_insert_idx = [6, 3]
            self.progressive_mask_nums = [0, 3, 7, 11]
        elif mode == 'XL-refine':
            cond_output_channels = [384] * 4 + [768] * 3 + [1536] * 6
            project_channels = [192] * 4 + [384] * 3 + [768] * 6
            concat_channels = [384] * 2 + [768] * 3 + [1536] * 7 + [0]
            cross_attn_insert_idx = [9, 6, 3]
            self.progressive_mask_nums = [0, 3, 6, 10, 14]
        else:
            raise NotImplementedError

        project_channels = [int(c * project_channel_scale) for c in project_channels]

        self.project_modules = nn.ModuleList()
        for i in range(len(cond_output_channels)):
            # if i == len(cond_output_channels) - 1:
            #     _project_type = 'ZeroCrossAttn'
            # else:
            #     _project_type = project_type
            _project_type = project_type
            if _project_type == 'ZeroSFT':
                self.project_modules.append(ZeroSFT(project_channels[i], cond_output_channels[i],
                                                    concat_channels=concat_channels[i]))
            elif _project_type == 'ZeroCrossAttn':
                self.project_modules.append(ZeroCrossAttn(cond_output_channels[i], project_channels[i]))
            else:
                raise NotImplementedError

        for i in cross_attn_insert_idx:
            self.project_modules.insert(i, ZeroCrossAttn(cond_output_channels[i], concat_channels[i]))
            # print(self.project_modules[i])

    def step_progressive_mask(self):
        if len(self.progressive_mask_nums) > 0:
            mask_num = self.progressive_mask_nums.pop()
            for i in range(len(self.project_modules)):
                if i < mask_num:
                    self.project_modules[i].mask = True
                else:
                    self.project_modules[i].mask = False
            return
            # print(f'step_progressive_mask, current masked layers: {mask_num}')
        else:
            return
            # print('step_progressive_mask, no more masked layers')
        # for i in range(len(self.project_modules)):
        #     print(self.project_modules[i].mask)


    def forward(self, x, timesteps=None, context=None, y=None, control=None, control_scale=1, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []

        _dtype = control[0].dtype
        x, context, y = x.to(_dtype), context.to(_dtype), y.to(_dtype)

        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype)
            emb = self.time_embed(t_emb)

            if self.num_classes is not None:
                assert y.shape[0] == x.shape[0]
                emb = emb + self.label_emb(y)

            # h = x.type(self.dtype)
            h = x
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)

        adapter_idx = len(self.project_modules) - 1
        control_idx = len(control) - 1
        h = self.middle_block(h, emb, context)
        h = self.project_modules[adapter_idx](control[control_idx], h, control_scale=control_scale)
        adapter_idx -= 1
        control_idx -= 1

        for i, module in enumerate(self.output_blocks):
            _h = hs.pop()
            h = self.project_modules[adapter_idx](control[control_idx], _h, h, control_scale=control_scale)
            adapter_idx -= 1
            # h = th.cat([h, _h], dim=1)
            if len(module) == 3:
                assert isinstance(module[2], Upsample)
                for layer in module[:2]:
                    if isinstance(layer, TimestepBlock):
                        h = layer(h, emb)
                    elif isinstance(layer, SpatialTransformer):
                        h = layer(h, context)
                    else:
                        h = layer(h)
                # print('cross_attn_here')
                h = self.project_modules[adapter_idx](control[control_idx], h, control_scale=control_scale)
                adapter_idx -= 1
                h = module[2](h)
            else:
                h = module(h, emb, context)
            control_idx -= 1
            # print(module)
            # print(h.shape)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            assert False, "not supported anymore. what the f*** are you doing?"
        else:
            return self.out(h)

if __name__ == '__main__':
    from omegaconf import OmegaConf

    # refiner
    # opt = OmegaConf.load('../../options/train/debug_p2_xl.yaml')
    #
    # model = instantiate_from_config(opt.model.params.control_stage_config)
    # hint = model(torch.randn([1, 4, 64, 64]), torch.randn([1]), torch.randn([1, 4, 64, 64]))
    # hint = [h.cuda() for h in hint]
    # print(sum(map(lambda hint: hint.numel(), model.parameters())))
    #
    # unet = instantiate_from_config(opt.model.params.network_config)
    # unet = unet.cuda()
    #
    # _output = unet(torch.randn([1, 4, 64, 64]).cuda(), torch.randn([1]).cuda(), torch.randn([1, 77, 1280]).cuda(),
    #                torch.randn([1, 2560]).cuda(), hint)
    # print(sum(map(lambda _output: _output.numel(), unet.parameters())))

    # base
    with torch.no_grad():
        opt = OmegaConf.load('../../options/dev/SUPIR_tmp.yaml')

        model = instantiate_from_config(opt.model.params.control_stage_config)
        model = model.cuda()

        hint = model(torch.randn([1, 4, 64, 64]).cuda(), torch.randn([1]).cuda(), torch.randn([1, 4, 64, 64]).cuda(), torch.randn([1, 77, 2048]).cuda(),
                       torch.randn([1, 2816]).cuda())

        for h in hint:
            print(h.shape)
        #
        unet = instantiate_from_config(opt.model.params.network_config)
        unet = unet.cuda()
        _output = unet(torch.randn([1, 4, 64, 64]).cuda(), torch.randn([1]).cuda(), torch.randn([1, 77, 2048]).cuda(),
                       torch.randn([1, 2816]).cuda(), hint)


        # model = instantiate_from_config(opt.model.params.control_stage_config)
        # model = model.cuda()
        # # hint = model(torch.randn([1, 4, 64, 64]), torch.randn([1]), torch.randn([1, 4, 64, 64]))
        # hint = model(torch.randn([1, 4, 64, 64]).cuda(), torch.randn([1]).cuda(), torch.randn([1, 4, 64, 64]).cuda(), torch.randn([1, 77, 1280]).cuda(),
        #                torch.randn([1, 2560]).cuda())
        # # hint = [h.cuda() for h in hint]
        #
        # for h in hint:
        #     print(h.shape)
        #
        # unet = instantiate_from_config(opt.model.params.network_config)
        # unet = unet.cuda()
        # _output = unet(torch.randn([1, 4, 64, 64]).cuda(), torch.randn([1]).cuda(), torch.randn([1, 77, 1280]).cuda(),
        #                torch.randn([1, 2560]).cuda(), hint)
