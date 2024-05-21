#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Created :   2024/05/15 17:29:21
@Desc    :   Cleaned up batch inference template
@Ref     :   
'''
import time
import random
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import DPTForDepthEstimation

from model.video_diffusion.models.controlnet3d import ControlNet3DModel
from model.video_diffusion.models.unet_3d_condition import UNetPseudo3DConditionModel
from model.video_diffusion.pipelines.pipeline_stable_diffusion_controlnet3d import Controlnet3DStableDiffusionPipeline


POS_PROMPT = (
    " ,best quality, extremely detailed, HD, ultra, 8K, HQ, masterpiece, trending on artstation, art, smooth")
NEG_PROMPT = (
    "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, "
    "low quality, deformed body, bloated, ugly, blurry, low res, unaesthetic"
)

data_root = '/data/trc/videdit-benchmark/DynEdit'
foundation_model_root = '/mnt/CV_teamz/pretrained'
model_root = '/mnt/CV_550w_90T/users/turongcheng/swh/models'
method_name = 'vid2vid-zero'

config = OmegaConf.create(dict(
    data_root=data_root,
    config_file=f'{data_root}/config.yaml',
    output_dir=f'{data_root}/outputs/{method_name}',
    seed=33,
    # TODO define arguments
    control_net_path=f'{model_root}/controlavideo-depth',
    annotator_path = f'{model_root}/dpt-hybrid-midas',
    num_inference_steps=50,
    guidance_scale=7.5,
    video_scale=1.5,  # post processing smoothness
    init_noise_thres=0.1, # diffusions state initalization
))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # load model
    print('Loading models ...')
    device = torch.device('cuda')
    # TODO define model
    unet = UNetPseudo3DConditionModel.from_pretrained(
        config.control_net_path, torch_dtype = torch.float16, subfolder='unet',
    ).to("cuda") 
    controlnet = ControlNet3DModel.from_pretrained(
        config.control_net_path, torch_dtype = torch.float16, subfolder='controlnet',
    ).to("cuda")
    annotator_model = DPTForDepthEstimation.from_pretrained(config.annotator_path).to("cuda")

    video_controlnet_pipe = Controlnet3DStableDiffusionPipeline.from_pretrained(
        config.control_net_path, unet=unet, 
        controlnet=controlnet, annotator_model=annotator_model,
        torch_dtype = torch.float16,
    ).to("cuda")

    data_config = OmegaConf.load(config.config_file)
    set_seed(config.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)

    preprocess_elapsed_ls = []
    inference_elapsed_ls = []
    for row in tqdm(data_config['data']):
        output_dir = Path(f"{config.output_dir}/{row.video_id}")
        if output_dir.exists():
            print(f"Skip {row.video_id} ...")
            continue
        else:
            output_dir.mkdir(parents=True, exist_ok=True)

        # load video
        print(f"Processing {row.video_id} ...")
        video_path = f'{config.data_root}/videos/{row.video_id}.mp4'
        np_frames, fps_vid = video_controlnet_pipe.get_frames_preprocess(
            video_path, num_frames=8, sampling_rate=1, return_np=True)
        frames = torch.from_numpy(np_frames).div(255) * 2 - 1
        frames = rearrange(frames, "f h w c -> c f h w").unsqueeze(0)
        frames = rearrange(frames, 'b c f h w -> (b f) c h w')
        control_maps = video_controlnet_pipe.get_depth_map(frames, 512, 512, return_standard_norm=False)  # (b f) 1 h w

        control_maps = control_maps.to(dtype=controlnet.dtype, device=controlnet.device)
        control_maps = F.interpolate(control_maps, size=(512, 512), mode='bilinear', align_corners=False)
        control_maps = rearrange(control_maps, "(b f) c h w -> b c f h w", f=8)
        if control_maps.shape[1] == 1:
            control_maps = repeat(control_maps, 'b c f h w -> b (n c) f h w',  n=3)
        frames = torch.from_numpy(np_frames).div(255)
        frames = rearrange(frames, 'f h w c -> f c h w')
        v2v_input_frames =  torch.nn.functional.interpolate(
            frames, size=(512, 512), mode="bicubic", antialias=True)
        v2v_input_frames = rearrange(v2v_input_frames, '(b f) c h w -> b c f h w ', f=8)
        # TODO load video

        # # Optional
        # inverse_path = Path(f"{config.output_dir}/{row.video_id}/.cache")
        # inverse_path.mkdir(parents=True, exist_ok=True)
        
        # preprocess
        start = time.perf_counter()
        # TODO preprocess video
        preprocess_elapsed = time.perf_counter() - start
        preprocess_elapsed_ls.append(preprocess_elapsed)

        # edit
        print(f'Editting {row.video_id} ...')
        start = time.perf_counter()
        for i, edit in tqdm(enumerate(row.edit)):
            # TODO edit
            # prompts=edit['prompt'],
            # negative_prompts=edit['src_words']+negative_prompt,
            # inversion_prompt=row['prompt'],
            # edit['tgt_words']
            samples = []
            for i in range(8):
                sample = video_controlnet_pipe(
                    # controlnet_hint= control_maps[:,:,:each_sample_frame,:,:],
                    # images= v2v_input_frames[:,:,:each_sample_frame,:,:],
                    controlnet_hint=control_maps[:,:,i*8-1:(i+1)*8-1,:,:] if i>0 else control_maps[:,:,:8,:,:],
                    images=v2v_input_frames[:,:,i*8-1:(i+1)*8-1,:,:] if i>0 else v2v_input_frames[:,:,:8,:,:],
                    first_frame_output=samples[-1] if i>0 else None,
                    prompt=edit['prompt'],
                    num_inference_steps=config.num_inference_steps,
                    width=512, height=512,
                    guidance_scale=config.guidance_scale,
                    generator=generator,
                    video_scale=config.video_scale,  # per-frame as negative (>= 1 or set 0)
                    init_noise_by_residual_thres=config.init_noise_thres,
                    # residual-based init. larger thres ==> more smooth.
                    controlnet_conditioning_scale=1.0,
                    fix_first_frame=True,
                    in_domain=True, # whether to use the video model to generate the first frame.
                )[0][1:]
                samples.extend(sample)
            imageio.mimsave(output_dir/f'{i}.gif', samples, fps=4)

        inference_elapsed = time.perf_counter() - start
        inference_elapsed_ls.append(inference_elapsed)

    with open(f'{config.output_dir}/time.log', 'a') as f:
        f.write(f'Preprocess: {sum(preprocess_elapsed_ls)/len(preprocess_elapsed_ls):.2f} sec/video\n')
        n_prompts = len(row.edit)
        f.write(f'Edit:       {sum(inference_elapsed_ls)/len(inference_elapsed_ls)/n_prompts:.2f} sec/edit\n')
        f.write('Preprocess:\n')
        f.writelines([f'{e:.1f} ' for e in preprocess_elapsed_ls])
        f.write('\nEdit:\n')
        f.writelines([f'{e:.1f} ' for e in inference_elapsed_ls])
        f.write('\n')
    print('Everything done!')


if __name__ == '__main__':
    main()