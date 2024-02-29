import os
from pickle import TRUE

import gradio as gr
from gradio_imageslider import ImageSlider
import argparse
from SUPIR.util import HWC3, upscale_image, fix_resize, convert_dtype, Tensor2PIL
import numpy as np
import torch
from SUPIR.util import create_SUPIR_model, load_QF_ckpt
from PIL import Image
from llava.llava_agent import LLavaAgent
from CKPT_PTH import LLAVA_MODEL_PATH
import einops
import copy
import datetime
import time


parser = argparse.ArgumentParser()
parser.add_argument("--ip", type=str, default='127.0.0.1')
parser.add_argument("--share", type=str, default=False)
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--no_llava", action='store_true', default=False)
parser.add_argument("--use_image_slider", action='store_true', default=False)
parser.add_argument("--log_history", action='store_true', default=False)
parser.add_argument("--loading_half_params", action='store_true', default=False)
parser.add_argument("--use_tile_vae", action='store_true', default=False)
parser.add_argument("--encoder_tile_size", type=int, default=512)
parser.add_argument("--decoder_tile_size", type=int, default=64)
parser.add_argument("--load_8bit_llava", action='store_true', default=False)
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
model = create_SUPIR_model('options/SUPIR_v0.yaml', SUPIR_sign='Q')
if args.loading_half_params:
    model = model.half()
if args.use_tile_vae:
    model.init_tile_vae(encoder_tile_size=512, decoder_tile_size=64)
model = model.to(SUPIR_device)
model.first_stage_model.denoise_encoder_s1 = copy.deepcopy(model.first_stage_model.denoise_encoder)
model.current_model = 'v0-Q'
ckpt_Q, ckpt_F = load_QF_ckpt('options/SUPIR_v0.yaml')

# load LLaVA
if use_llava:
    llava_agent = LLavaAgent(LLAVA_MODEL_PATH, device=LLaVA_device, load_8bit=args.load_8bit_llava, load_4bit=False)
else:
    llava_agent = None


def stage1_process(input_image, gamma_correction):
    torch.cuda.set_device(SUPIR_device)
    LQ = HWC3(input_image)
    LQ = fix_resize(LQ, 512)
    # stage1
    LQ = np.array(LQ) / 255 * 2 - 1
    LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
    LQ = model.batchify_denoise(LQ, is_stage1=True)
    LQ = (LQ[0].permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().round().clip(0, 255).astype(np.uint8)
    # gamma correction
    LQ = LQ / 255.0
    LQ = np.power(LQ, gamma_correction)
    LQ *= 255.0
    LQ = LQ.round().clip(0, 255).astype(np.uint8)
    return LQ


def llava_process(input_image, temperature, top_p, qs=None):
    if use_llava:
        torch.cuda.set_device(LLaVA_device)
        LQ = HWC3(input_image)
        LQ = Image.fromarray(LQ.astype('uint8'))
        captions = llava_agent.gen_image_caption([LQ], temperature=temperature, top_p=top_p, qs=qs)
    else:
        captions = ['LLaVA is not available. Please add text manually.']
    return captions[0]


def batch_upscale(batch_process_folder, outputs_folder, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps,
                  s_stage1, s_stage2, s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype,
                  gamma_correction, linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select,
                  num_images, random_seed, progress=gr.Progress()):
    import os
    import numpy as np
    from PIL import Image

    # Get the list of image files in the folder
    image_files = [file for file in os.listdir(batch_process_folder) if file.lower().endswith((".png", ".jpg", ".jpeg"))]
    total_images = len(image_files)
    main_prompt = prompt
    # Iterate over all image files in the folder
    for index, file_name in enumerate(image_files):
        try:
            progress((index + 1) / total_images, f"Processing {index + 1}/{total_images} image")
            # Construct the full file path
            file_path = os.path.join(batch_process_folder, file_name)
            prompt = main_prompt
            # Open the image file and convert it to a NumPy array
            with Image.open(file_path) as img:
                img_array = np.asarray(img)

            # Construct the path for the prompt text file
            base_name = os.path.splitext(file_name)[0]
            prompt_file_path = os.path.join(batch_process_folder, f"{base_name}.txt")

            # Read the prompt from the text file
            if os.path.exists(prompt_file_path):
                with open(prompt_file_path, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()

            # Call the stage2_process method for the image
            stage2_process(img_array, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                           s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                           linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select, num_images,
                           random_seed, dont_update_progress=True, outputs_folder=outputs_folder)

            # Update progress
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    return "All Done"


def stage2_process(input_image, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                   s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                   linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select, num_images,
                   random_seed, dont_update_progress=False, outputs_folder="outputs", progress=gr.Progress()):

    torch.cuda.set_device(SUPIR_device)

    event_id = str(time.time_ns())
    event_dict = {'event_id': event_id, 'localtime': time.ctime(), 'prompt': prompt, 'a_prompt': a_prompt,
                  'n_prompt': n_prompt, 'num_samples': num_samples, 'upscale': upscale, 'edm_steps': edm_steps,
                  's_stage1': s_stage1, 's_stage2': s_stage2, 's_cfg': s_cfg, 'seed': seed, 's_churn': s_churn,
                  's_noise': s_noise, 'color_fix_type': color_fix_type, 'diff_dtype': diff_dtype, 'ae_dtype': ae_dtype,
                  'gamma_correction': gamma_correction, 'linear_CFG': linear_CFG, 'linear_s_stage2': linear_s_stage2,
                  'spt_linear_CFG': spt_linear_CFG, 'spt_linear_s_stage2': spt_linear_s_stage2,
                  'model_select': model_select}

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
    input_image = upscale_image(input_image, upscale, unit_resolution=32, min_size=1024)

    LQ = np.array(input_image) / 255.0
    LQ = np.power(LQ, gamma_correction)
    LQ *= 255.0
    LQ = LQ.round().clip(0, 255).astype(np.uint8)
    LQ = LQ / 255 * 2 - 1
    LQ = torch.tensor(LQ, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(SUPIR_device)[:, :3, :, :]
    captions = [prompt]

    model.ae_dtype = convert_dtype(ae_dtype)
    model.model.dtype = convert_dtype(diff_dtype)

    output_dir = os.path.join("outputs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if outputs_folder.strip() != "" and outputs_folder != "outputs":
        output_dir = outputs_folder
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    all_results = []
    counter = 1
    if not dont_update_progress:
        progress(0 / num_images, desc="Generating images")
    for _ in range(num_images):
        if random_seed or num_images>1:
            seed = np.random.randint(0, 2147483647)
        start_time = time.time()  # Track the start time
        samples = model.batchify_sample(LQ, captions, num_steps=edm_steps, restoration_scale=s_stage1, s_churn=s_churn,
                                        s_noise=s_noise, cfg_scale=s_cfg, control_scale=s_stage2, seed=seed,
                                        num_samples=num_samples, p_p=a_prompt, n_p=n_prompt, color_fix_type=color_fix_type,
                                        use_linear_CFG=linear_CFG, use_linear_control_scale=linear_s_stage2,
                                        cfg_scale_start=spt_linear_CFG, control_scale_start=spt_linear_s_stage2)

        x_samples = (einops.rearrange(samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().round().clip(
            0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(num_samples)]
        image_generation_time = time.time() - start_time
        desc = f"Generated image {counter}/{num_images} in {image_generation_time:.2f} seconds"
        counter = counter+1
        if not dont_update_progress:
            progress(counter / num_images, desc=desc)
        print(desc)  # Print the progress
        start_time = time.time()  # Reset the start time for the next image

        for i, result in enumerate(results):
            timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            save_path = os.path.join(output_dir, f'{timestamp}.png')
            Image.fromarray(result).save(save_path)
        all_results.extend(results)

    if args.log_history:
        os.makedirs(f'./history/{event_id[:5]}/{event_id[5:]}', exist_ok=True)
        with open(f'./history/{event_id[:5]}/{event_id[5:]}/logs.txt', 'w') as f:
            f.write(str(event_dict))
        f.close()
        Image.fromarray(input_image).save(f'./history/{event_id[:5]}/{event_id[5:]}/LQ.png')
        for i, result in enumerate(all_results):
            Image.fromarray(result).save(f'./history/{event_id[:5]}/{event_id[5:]}/HQ_{i}.png')
    return [input_image] + all_results, event_id, 3, '', seed


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
    linear_s_stage2 = False
    if param_setting == "Quality":
        s_cfg = 7.5
        linear_CFG = False
    elif param_setting == "Fidelity":
        s_cfg = 4.0
        linear_CFG = True
    else:
        raise NotImplementedError
    return edm_steps, s_cfg, s_stage2, s_stage1, s_churn, s_noise, a_prompt, n_prompt, color_fix_type, linear_CFG, \
        linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2


def submit_feedback(event_id, fb_score, fb_text):
    if args.log_history:
        with open(f'./history/{event_id[:5]}/{event_id[5:]}/logs.txt', 'r') as f:
            event_dict = eval(f.read())
        f.close()
        event_dict['feedback'] = {'score': fb_score, 'text': fb_text}
        with open(f'./history/{event_id[:5]}/{event_id[5:]}/logs.txt', 'w') as f:
            f.write(str(event_dict))
        f.close()
        return 'Submit successfully, thank you for your comments!'
    else:
        return 'Submit failed, the server is not set to log history.'


title_md = """
# **SUPIR: Practicing Model Scaling for Photo-Realistic Image Restoration**

⚠️SUPIR is still a research project under tested and is not yet a stable commercial product.

[[Paper](https://arxiv.org/abs/2401.13627)] &emsp; [[Project Page](http://supir.xpixel.group/)] &emsp; [[How to play](https://github.com/Fanghua-Yu/SUPIR/blob/master/assets/DemoGuide.png)]
"""


claim_md = """
## **Terms of use**

By using this service, users are required to agree to the following terms: The service is a research preview intended for non-commercial use only. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. The service may collect user dialogue data for future research. Please submit a feedback to us if you get any inappropriate answer! We will collect those to keep improving our models. For an optimal experience, please use desktop computers for this demo, as mobile devices may compromise its quality.

## **License**

The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/Fanghua-Yu/SUPIR) of SUPIR.
"""


block = gr.Blocks(title='SUPIR').queue()
with block:
    with gr.Row():
        gr.Markdown(title_md)
    with gr.Row():
        with gr.Column():
            with gr.Row(equal_height=True):
                with gr.Column():
                    gr.Markdown("<center>Input</center>")
                    input_image = gr.Image(type="numpy", elem_id="image-input", height=400, width=400)
                with gr.Column():
                    gr.Markdown("<center>Stage1 Output</center>")
                    denoise_image = gr.Image(type="numpy", elem_id="image-s1", height=400, width=400)
            prompt = gr.Textbox(label="Prompt", value="")
            with gr.Accordion("Stage1 options", open=False):
                gamma_correction = gr.Slider(label="Gamma Correction", minimum=0.1, maximum=2.0, value=1.0, step=0.1)

            with gr.Accordion("Stage2 options", open=True):
               with gr.Row():
                  with gr.Column():
                        num_images = gr.Slider(label="Number Of Images To Generate", minimum=1, maximum=200
                                        , value=1, step=1)
                        num_samples = gr.Slider(label="Batch Size", minimum=1, maximum=4 if not args.use_image_slider else 1
                                        , value=1, step=1)
                  with gr.Column():
                    upscale = gr.Slider(label="Upscale", minimum=1, maximum=8, value=1, step=0.1)
                    random_seed = gr.Checkbox(label="Randomize Seed", value=True)
               with gr.Row():
                    edm_steps = gr.Slider(label="Steps", minimum=20, maximum=200, value=50, step=1)
                    s_cfg = gr.Slider(label="Text Guidance Scale", minimum=1.0, maximum=15.0, value=7.5, step=0.1)
                    s_stage2 = gr.Slider(label="Stage2 Guidance Strength", minimum=0., maximum=1., value=1., step=0.05)
                    s_stage1 = gr.Slider(label="Stage1 Guidance Strength", minimum=-1.0, maximum=6.0, value=-1.0, step=1.0)
                    seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                    s_churn = gr.Slider(label="S-Churn", minimum=0, maximum=40, value=5, step=1)
                    s_noise = gr.Slider(label="S-Noise", minimum=1.0, maximum=1.1, value=1.003, step=0.001)
               with gr.Row():
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

        with gr.Column():
            gr.Markdown("<center>Upscaled Images Output</center>")
            if not args.use_image_slider:
                result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery1")
            else:
                result_gallery = ImageSlider(label='Output', show_label=False, elem_id="gallery1")
            with gr.Row():
                with gr.Column():
                    denoise_button = gr.Button(value="Stage1 Run")
                with gr.Column():
                    llava_button = gr.Button(value="LlaVa Run")
                with gr.Column():
                    diffusion_button = gr.Button(value="Stage2 Run")
            with gr.Row():
                with gr.Column():
                    batch_process_folder = gr.Textbox(label="Batch Processing Input Folder Path - If image_file_name.txt exists it will be read and used as prompt (optional). Uses same settings of single upscale (Stage 2 Run). If no caption txt it will use the Prompt you written. It can be empty as well.", placeholder="e.g. R:\SUPIR video\comparison_images")
                    outputs_folder = gr.Textbox(label="Batch Processing Output Folder Path - If left empty images are saved in default folder", placeholder="e.g. R:\SUPIR video\comparison_images\outputs")
            with gr.Row():
                with gr.Column():
                    batch_upscale_button = gr.Button(value="Start Batch Upscaling")
                    outputlabel = gr.Label("Batch Processing Progress")
            with gr.Row():
                with gr.Column():
                    param_setting = gr.Dropdown(["Quality", "Fidelity"], interactive=True, label="Param Setting",
                                               value="Quality")
                with gr.Column():
                    restart_button = gr.Button(value="Reset Param", scale=2)
            with gr.Row():
                with gr.Column():
                    linear_CFG = gr.Checkbox(label="Linear CFG", value=False)
                    spt_linear_CFG = gr.Slider(label="CFG Start", minimum=1.0,
                                                    maximum=9.0, value=1.0, step=0.5)
                with gr.Column():
                    linear_s_stage2 = gr.Checkbox(label="Linear Stage2 Guidance", value=False)
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
            with gr.Accordion("LLaVA options", open=False):
                temperature = gr.Slider(label="Temperature", minimum=0., maximum=1.0, value=0.2, step=0.1)
                top_p = gr.Slider(label="Top P", minimum=0., maximum=1.0, value=0.7, step=0.1)
                qs = gr.Textbox(label="Question", value="Describe this image and its style in a very detailed manner. "
                                                        "The image is a realistic photography, not an art painting.")
            with gr.Accordion("Feedback", open=False):
                fb_score = gr.Slider(label="Feedback Score", minimum=1, maximum=5, value=3, step=1,
                                     interactive=True)
                fb_text = gr.Textbox(label="Feedback Text", value="", placeholder='Please enter your feedback here.')
                submit_button = gr.Button(value="Submit Feedback")
    with gr.Row():
        gr.Markdown(claim_md)
        event_id = gr.Textbox(label="Event ID", value="", visible=False)

    llava_button.click(fn=llava_process, inputs=[denoise_image, temperature, top_p, qs], outputs=[prompt])
    denoise_button.click(fn=stage1_process, inputs=[input_image, gamma_correction],
                         outputs=[denoise_image])
    stage2_ips = [input_image, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                  s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                  linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select,num_images,random_seed]
    diffusion_button.click(fn=stage2_process, inputs=stage2_ips, outputs=[result_gallery, event_id, fb_score, fb_text, seed], show_progress=True, queue=True)
    restart_button.click(fn=load_and_reset, inputs=[param_setting],
                         outputs=[edm_steps, s_cfg, s_stage2, s_stage1, s_churn, s_noise, a_prompt, n_prompt,
                                  color_fix_type, linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2])
    submit_button.click(fn=submit_feedback, inputs=[event_id, fb_score, fb_text], outputs=[fb_text])
    stage2_ips_batch = [batch_process_folder,outputs_folder, prompt, a_prompt, n_prompt, num_samples, upscale, edm_steps, s_stage1, s_stage2,
                  s_cfg, seed, s_churn, s_noise, color_fix_type, diff_dtype, ae_dtype, gamma_correction,
                  linear_CFG, linear_s_stage2, spt_linear_CFG, spt_linear_s_stage2, model_select,num_images,random_seed]
    batch_upscale_button.click(fn=batch_upscale, inputs=stage2_ips_batch, outputs=outputlabel, show_progress=True, queue=True)
block.launch(server_name=server_ip, server_port=server_port, share=args.share, inbrowser=True)
