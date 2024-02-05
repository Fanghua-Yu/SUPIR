import gradio as gr
from gradio_imageslider import ImageSlider
import argparse
from SUPIR.util import HWC3, upscale_image, fix_resize, convert_dtype
import numpy as np
import torch
from SUPIR.util import create_SUPIR_model, load_QF_ckpt
from PIL import Image
from llava.llava_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH
import einops

parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, default='6688')
parser.add_argument("--no_llava", action='store_true', default=False)
parser.add_argument("--use_image_slider", action='store_true', default=False)
args = parser.parse_args()
server_ip = args.ip
server_port = args.port
use_llava = not args.no_llava

if torch.cuda.device_count() >= 2:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:1'
elif torch.cuda.device_count() == 1:
    SUPIR_device = 'cuda:0'
    LLaVA_device = 'cuda:0'
else:
    raise ValueError('Currently support CUDA only.')

# load SUPIR
model = create_SUPIR_model('options/SUPIR_v0.yaml', SUPIR_sign='Q').to(SUPIR_device)
model.current_model = 'v0-Q'
ckpt_Q, ckpt_F = load_QF_ckpt('options/SUPIR_v0.yaml')
# load LLaVA
if use_llava:
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=LLaVA_device)
else:
    llava_agent = None


def stage1_process(input_image, gamma_correction):
    # force to v0-Q
    if model.current_model != 'v0-Q':
        print('load v0-Q')
        model.load_state_dict(ckpt_Q, strict=False)
        model.current_model = 'v0-Q'
    LQ = HWC3(input_image)
    LQ = fix_resize(LQ, 512)
    # stage1
    LQ = np.array(LQ) / 255 * 2 - 1
    LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
    LQ = model.batchify_denoise(LQ)
    LQ = (LQ[0].permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(np.uint8)
    # gamma correction
    LQ = LQ / 255.0
    LQ = np.power(LQ, gamma_correction)
    LQ *= 255.0
    LQ = LQ.round().clip(0, 255).astype(np.uint8)
    return LQ

def llave_process(input_image, temperature, top_p, qs=None):
    if use_llava:
        LQ = HWC3(input_image)
        LQ = Image.fromarray(LQ.astype('uint8'))
        captions = llava_agent.gen_image_caption([LQ], temperature=temperature, top_p=top_p, qs=qs)
    else:
        captions = ['LLaVA is not available. Please add text manually.']
    return captions[0]

def stage2_process(input_image, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                   s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                   linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select):
    if model_select != model.current_model:
        if model_select == 'v0-Q':
            print('load v0-Q')
            model.load_state_dict(ckpt_Q, strict=False)
            model.current_model = 'v0-Q'
        elif model_select == 'v0-F':
            print('load v0-F')
            model.load_state_dict(ckpt_F, strict=False)
            model.current_model = 'v0-F'
    input_image = HWC3(input_image)
    input_image = upscale_image(input_image, upscale, unit_resolution=32)

    LQ = np.array(input_image) / 255.0
    LQ = np.power(LQ, gamma_correction)
    LQ *= 255.0
    LQ = LQ.round().clip(0, 255).astype(np.uint8)
    LQ = LQ / 255 * 2 - 1
    LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
    captions = [prompt]

    model.ae_dtype = convert_dtype(ae_dtype)
    model.model.dtype = convert_dtype(diff_dtype)

    samples = model.batchify_sample(LQ, captions, num_steps=edm_steps, restoration_scale=s_stage1, s_churn=s_churn,
                                    s_noise=s_noise, cfg_scale=s_cfg, control_scale=s_stage2, seed=seed,
                                    num_samples=num_samples, p_p=a_prompt, n_p=n_prompt, color_fix_type=color_fix_type,
                                    use_linear_CFG=linear_CFG, use_linear_control_scale=linear_s_stage2,
                                    cfg_scale_start=spt_linear_CFG, control_scale_start=spt_linear_s_stage2)

    x_samples = (einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(
        0, 255).astype(np.uint8)
    results = [x_samples[i] for i in range(num_samples)]
    return [input_image] + results

def load_and_reset(param_setting):
    edm_steps = 50
    s_stage2 = 1.0
    s_stage1 = -1.0
    s_churn = 5
    s_noise = 1.003
    a_prompt = 'Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, hyper detailed photo - ' \
               'realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, skin pore ' \
               'detailing, hyper sharpness, perfect without deformations.'
    n_prompt = 'painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, CG Style, ' \
               '3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, ' \
               'signature, jpeg artifacts, deformed, lowres, over-smooth'
    color_fix_type = 'Wavelet'
    spt_linear_CFG = 1.0
    spt_linear_s_stage2 = 0.0
    if param_setting == "Quality":
        s_cfg = 7.5
        linear_CFG = False
        linear_s_stage2 = True
    elif param_setting == "Fidelity":
        s_cfg = 4.0
        linear_CFG = True
        linear_s_stage2 = False
    else:
        raise NotImplementedError
    return edm_steps, s_cfg, s_stage2, s_stage1, s_churn, s_noise, a_prompt, n_prompt, color_fix_type, linear_CFG, \
        linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2

block = gr.Blocks(title='SUPIR').queue()
with block:
    with gr.Row():
        gr.Markdown("<center><font size=5>SUPIR Playground</font></center>")
    with gr.Row():
        with gr.Column():
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("<center>Input</center>")
                    input_image = gr.Image(type="numpy", elem_id="image-input")
                with gr.Column():
                    gr.Markdown("<center>Stage1 Output</center>")
                    denoise_image = gr.Image(type="numpy", elem_id="image-s1")
            prompt = gr.Textbox(label="Prompt", value="")
            with gr.Accordion("Stage1 options", open=False):
                gamma_correction = gr.Slider(label="Gamma Correction", minimum=0.1, maximum=2.0, value=1.0, step=0.1)
            with gr.Accordion("LLaVA options", open=False):
                temperature = gr.Slider(label="Temperature", minimum=0., maximum=1.0, value=0.2, step=0.1)
                top_p = gr.Slider(label="Top P", minimum=0., maximum=1.0, value=0.7, step=0.1)
                qs = gr.Textbox(label="Question", value="Describe this image and its style in a very detailed manner.")
            with gr.Accordion("Stage2 options", open=False):
                num_samples = gr.Slider(label="Num Samples", minimum=1, maximum=4 if not args.use_image_slider else 1
                                        , value=1, step=1)
                upscale = gr.Slider(label="Upscale", minimum=1, maximum=8, value=1, step=1)
                edm_steps = gr.Slider(label="Steps", minimum=20, maximum=200, value=50, step=1)
                s_cfg = gr.Slider(label="Text Guidance Scale", minimum=1.0, maximum=15.0, value=7.5, step=0.1)
                s_stage2 = gr.Slider(label="Stage2 Guidance Strength", minimum=0., maximum=1., value=1., step=0.05)
                s_stage1 = gr.Slider(label="Stage1 Guidance Strength", minimum=-1.0, maximum=6.0, value=-1.0, step=1.0)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                s_churn = gr.Slider(label="S-Churn", minimum=0, maximum=40, value=5, step=1)
                s_noise = gr.Slider(label="S-Noise", minimum=1.0, maximum=1.1, value=1.003, step=0.001)
                a_prompt = gr.Textbox(label="Default Positive Prompt",
                                      value='Cinematic, High Contrast, highly detailed, taken using a Canon EOS R '
                                            'camera, hyper detailed photo - realistic maximum detail, 32k, Color '
                                            'Grading, ultra HD, extreme meticulous detailing, skin pore detailing, '
                                            'hyper sharpness, perfect without deformations.')
                n_prompt = gr.Textbox(label="Default Negative Prompt",
                                      value='painting, oil painting, illustration, drawing, art, sketch, oil painting, '
                                            'cartoon, CG Style, 3D render, unreal engine, blurring, dirty, messy, '
                                            'worst quality, low quality, frames, watermark, signature, jpeg artifacts, '
                                            'deformed, lowres, over-smooth')
                with gr.Row():
                    with gr.Column():
                        linear_CFG = gr.Checkbox(label="Linear CFG", value=False)
                        spt_linear_CFG = gr.Slider(label="CFG Start", minimum=1.0,
                                                        maximum=9.0, value=1.0, step=0.5)
                    with gr.Column():
                        linear_s_stage2 = gr.Checkbox(label="Linear Stage2 Guidance", value=True)
                        spt_linear_s_stage2 = gr.Slider(label="Guidance Start", minimum=0.,
                                                        maximum=1., value=0., step=0.05)
                with gr.Row():
                    with gr.Column():
                        diff_dtype = gr.Radio(['fp32', 'fp16', 'bf16'], label="Diffusion Data Type", value="fp16",
                                              interactive=True)
                    with gr.Column():
                        ae_dtype = gr.Radio(['fp32', 'bf16'], label="Auto-Encoder Data Type", value="bf16",
                                            interactive=True)
                    with gr.Column():
                        color_fix_type = gr.Radio(["None", "AdaIn", "Wavelet"], label="Color-Fix Type", value="Wavelet",
                                                  interactive=True)
                    with gr.Column():
                        model_select = gr.Radio(["v0-Q", "v0-F"], label="Model Selection", value="v0-Q",
                                                interactive=True)

        with gr.Column():
            gr.Markdown("<center>Stage2 Output</center>")
            if not args.use_image_slider:
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery1")
            else:
                result_gallery = ImageSlider(label='Output', show_label=False, elem_id="gallery1")
            with gr.Row():
                with gr.Column():
                    denoise_button = gr.Button(value="Stage1 Run")
                with gr.Column():
                    llave_button = gr.Button(value="LlaVa Run")
                with gr.Column():
                    diffusion_button = gr.Button(value="Stage2 Run")
            with gr.Row():
                with gr.Column():
                    param_setting = gr.Dropdown(["Quality", "Fidelity"], interactive=True, label="Param Setting",
                                               value="Quality")
                with gr.Column():
                    restart_button = gr.Button(value="Reset Param")


    llave_button.click(fn=llave_process, inputs=[denoise_image, temperature, top_p, qs], outputs=[prompt])
    denoise_button.click(fn=stage1_process, inputs=[input_image, gamma_correction],
                         outputs=[denoise_image])
    stage2_ips = [input_image, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                  s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                  linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select]
    diffusion_button.click(fn=stage2_process, inputs=stage2_ips, outputs=[result_gallery])
    restart_button.click(fn=load_and_reset, inputs=[param_setting],
                         outputs=[edm_steps, s_cfg, s_stage2, s_stage1, s_churn, s_noise, a_prompt, n_prompt,
                                  color_fix_type, linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2])
block.launch(server_name=server_ip, server_port=server_port)
