from typing import Any, Dict

from diffusers import AutoencoderKLTemporalDecoder
from einops import rearrange
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

from video_to_video.diffusion.diffusion_sdedit import GaussianDiffusion
from video_to_video.diffusion.schedules_sdedit import noise_schedule
from video_to_video.modules.embedder import FrozenOpenCLIPEmbedder
import video_to_video.modules.unet_v2v as unet_v2v
from video_to_video.utils.config import cfg
from video_to_video.utils.logger import get_logger
from video_to_video.utils.util import *
from easydict import EasyDict
import os
from huggingface_hub import hf_hub_download
from video_to_video.video_to_video_model import VideoToVideo
import cv2
import tempfile
import subprocess


def download_model(version="v2"):
    REPO_ID = "jwhejwhe/VEnhancer"
    filename = "venhancer_paper.pt"
    if version == "v2":
        filename = "venhancer_v2.pt"
    ckpt_dir = "./ckpts/"
    os.makedirs(ckpt_dir, exist_ok=True)
    local_file = os.path.join(ckpt_dir, filename)
    if not os.path.exists(local_file):
        print(f"Downloading the VEnhancer checkpoint...")
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=ckpt_dir)
    return local_file

model_path = download_model()
model_cfg = EasyDict(__name__="model_cfg")
model_cfg.model_path = model_path
model = VideoToVideo(model_cfg)

gen_vid = torch.load("gen_vid.pt")

rolled_gen_vid = gen_vid.roll(gen_vid.shape[-1] // 2, dims=-1)

# blur the area in the middle of the width
pad_width = 2
rolled_gen_vid_mid = rolled_gen_vid[:, :, :, :, gen_vid.shape[-1] // 2 - pad_width:gen_vid.shape[-1] // 2 + pad_width]
rolled_gen_vid_mid = F.avg_pool2d(rolled_gen_vid_mid.flatten(0, 1), kernel_size=3, stride=1, padding=1).view_as(rolled_gen_vid_mid)
rolled_gen_vid[:, :, :, :, gen_vid.shape[-1] // 2 - pad_width:gen_vid.shape[-1] // 2 + pad_width] = rolled_gen_vid_mid

decoded_vid = model.tiled_chunked_decode(rolled_gen_vid)
# decoded_vid = decoded_vid.roll(-decoded_vid.shape[-1] // 2, dims=-1) 

# import pdb; pdb.set_trace()

def tensor2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)
    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    video = video * 255.0
    images = rearrange(video, "b c f h w -> b f h w c")[0]
    return images

def save_video(video, save_dir, file_name, fps=16.0, drop_prefix_cnt=0):
    output_path = os.path.join(save_dir, file_name)
    images = [(img.numpy()).astype("uint8") for img in video]
    temp_dir = tempfile.mkdtemp()
    for fid, frame in enumerate(images[drop_prefix_cnt:]):
        tpth = os.path.join(temp_dir, "%06d.png" % (fid + 1))
        cv2.imwrite(tpth, frame[:, :, ::-1])
    tmp_path = os.path.join(save_dir, "tmp.mp4")
    cmd = f"ffmpeg -y -f image2 -framerate {fps} -i {temp_dir}/%06d.png \
     -vcodec libx264 -crf 17 -pix_fmt yuv420p {tmp_path}"
    status, output = subprocess.getstatusoutput(cmd)
    if status != 0:
        print(f"Save Video Error with {output}")
    os.system(f"rm -rf {temp_dir}")
    os.rename(tmp_path, output_path)


decoded_vid = tensor2vid(decoded_vid[:, :, :, :-16, :]).cpu()

# use ffmpeg to save the video
import subprocess
fps = 5

# save the video
save_video(decoded_vid, ".", "decoded_vid_blurred.mp4", fps=fps)
