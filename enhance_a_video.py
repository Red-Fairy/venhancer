from argparse import ArgumentParser, Namespace
import glob

from easydict import EasyDict
from huggingface_hub import hf_hub_download

from inference_utils import *
from video_to_video.utils.seed import setup_seed
from video_to_video.video_to_video_model import VideoToVideo

logger = get_logger()


class VEnhancer:
    def __init__(
        self,
        result_dir="./results/",
        version="v1",
        model_path="",
        solver_mode="fast",
        steps=15,
        guide_scale=7.5,
        s_cond=8,
    ):
        if not model_path:
            self.download_model(version=version)
        else:
            self.model_path = model_path
        assert os.path.exists(self.model_path), "Error: checkpoint Not Found!"
        logger.info(f"checkpoint_path: {self.model_path}")

        self.result_dir = result_dir
        os.makedirs(self.result_dir, exist_ok=True)

        model_cfg = EasyDict(__name__="model_cfg")
        model_cfg.model_path = self.model_path
        self.model = VideoToVideo(model_cfg)

        steps = 15 if solver_mode == "fast" else steps
        self.solver_mode = solver_mode
        self.steps = steps
        self.guide_scale = guide_scale
        self.s_cond = s_cond

    def enhance_a_video(self, video_path, prompt, up_scale=4, target_fps=24, noise_aug=300, blended_decoding=False, rotation_decoding=False):

        save_name = os.path.splitext(os.path.basename(video_path))[0]
        text = prompt
        logger.info(f"text: {text}")
        caption = text + self.model.positive_prompt

        input_frames, input_fps = load_video(video_path)
        in_f_num = len(input_frames)
        logger.info(f"input frames length: {in_f_num}")
        logger.info(f"input fps: {input_fps}")
        interp_f_num = max(round(target_fps / input_fps) - 1, 0)
        interp_f_num = min(interp_f_num, 8)
        target_fps = input_fps * (interp_f_num + 1)
        logger.info(f"target_fps: {target_fps}")

        video_data = preprocess(input_frames)
        _, _, h, w = video_data.shape
        logger.info(f"input resolution: {(h, w)}")
        target_h, target_w = adjust_resolution(h, w, up_scale)
        logger.info(f"target resolution: {(target_h, target_w)}")

        mask_cond = make_mask_cond(in_f_num, interp_f_num)
        mask_cond = torch.Tensor(mask_cond).long()

        noise_aug = min(max(noise_aug, 0), 300)
        logger.info(f"noise augmentation: {noise_aug}")
        logger.info(f"scale s is set to: {self.s_cond}")

        pre_data = {"video_data": video_data, "y": caption}
        pre_data["mask_cond"] = mask_cond
        pre_data["s_cond"] = self.s_cond
        pre_data["interp_f_num"] = interp_f_num
        pre_data["target_res"] = (target_h, target_w)
        pre_data["t_hint"] = noise_aug

        total_noise_levels = 900
        setup_seed(666)

        with torch.no_grad():
            data_tensor = collate_fn(pre_data, "cuda:0")
            output = self.model.test(
                data_tensor,
                total_noise_levels,
                steps=self.steps,
                solver_mode=self.solver_mode,
                guide_scale=self.guide_scale,
                noise_aug=noise_aug,
                blended_decoding=blended_decoding,
                rotation_decoding=rotation_decoding,
            )

        output = tensor2vid(output)
        save_video(output, self.result_dir, f"{save_name}_enhanced.mp4", fps=target_fps)
        return os.path.join(self.result_dir, f"{save_name}_enhanced.mp4")

    def download_model(self, version="v1"):
        REPO_ID = "jwhejwhe/VEnhancer"
        filename = "venhancer_paper.pt"
        if version == "v2":
            filename = "venhancer_v2.pt"
        ckpt_dir = "./ckpts/"
        os.makedirs(ckpt_dir, exist_ok=True)
        local_file = os.path.join(ckpt_dir, filename)
        if not os.path.exists(local_file):
            logger.info(f"Downloading the VEnhancer checkpoint...")
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=ckpt_dir)
        self.model_path = local_file


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--input_path", required=True, type=str, help="input video path")
    parser.add_argument("--save_dir", type=str, default="results", help="save directory")
    parser.add_argument("--version", type=str, default="v1", choices=["v1", "v2"], help="model version")
    parser.add_argument("--model_path", type=str, default="", help="model path")

    parser.add_argument("--blended_decoding", action="store_true", default=False)
    parser.add_argument("--rotation_decoding", action="store_true", default=False)

    parser.add_argument("--prompt", type=str, default="a good video", help="prompt")
    parser.add_argument("--prompt_path", type=str, default="", help="prompt path")
    parser.add_argument("--filename_as_prompt", action="store_true")

    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--solver_mode", type=str, default="fast", choices=["fast", "normal"], help="fast | normal")
    parser.add_argument("--steps", type=int, default=15)

    parser.add_argument("--noise_aug", type=int, default=200, help="noise augmentation")
    parser.add_argument("--target_fps", type=int, default=24)
    parser.add_argument("--up_scale", type=float, default=4)
    parser.add_argument("--s_cond", type=float, default=8)

    return parser.parse_args()


def main():

    args = parse_args()

    input_path = args.input_path
    prompt = args.prompt
    prompt_path = args.prompt_path
    filename_as_prompt = args.filename_as_prompt
    model_path = args.model_path
    version = args.version
    save_dir = args.save_dir

    noise_aug = args.noise_aug
    up_scale = args.up_scale
    target_fps = args.target_fps
    s_cond = args.s_cond

    steps = args.steps
    solver_mode = args.solver_mode
    guide_scale = args.cfg
    blended_decoding = args.blended_decoding
    rotation_decoding = args.rotation_decoding

    venhancer = VEnhancer(
        result_dir=save_dir,
        version=version,
        model_path=model_path,
        solver_mode=solver_mode,
        steps=steps,
        guide_scale=guide_scale,
        s_cond=s_cond,
    )

    if os.path.isdir(input_path):
        file_path_list = sorted(glob.glob(os.path.join(input_path, "*.mp4")))
    elif os.path.isfile(input_path):
        file_path_list = [input_path]
    else:
        raise TypeError("input must be a directory or video file!")

    prompt_list = None
    if os.path.isfile(prompt_path):
        prompt_list = load_prompt_list(prompt_path)
        if len(prompt_list) != len(file_path_list):
            if len(prompt_list) == 1:
                prompt_list = prompt_list * len(file_path_list)
            else:
                raise ValueError("prompt_list length must be equal to file_path_list length!")
    else:
        prompt_list = [prompt_path] * len(file_path_list)

    for ind, file_path in enumerate(file_path_list):
        prompt = prompt_list[ind]
        logger.info(f"processing video {ind}, file_path: {file_path}")
        # if filename_as_prompt:
        #     prompt = os.path.splitext(os.path.basename(file_path))[0]
        # elif prompt_list is not None:
        #     prompt = prompt_list[ind]
        # else:
        #     prompt_path = os.path.splitext(file_path)[0] + ".txt"
        #     if os.path.isfile(prompt_path):
        #         logger.info(f"prompt_path: {prompt_path}")
        #         prompt = load_prompt_list(prompt_path)[0]
        venhancer.enhance_a_video(file_path, prompt, up_scale, target_fps, noise_aug, blended_decoding, rotation_decoding)


if __name__ == "__main__":
    main()
