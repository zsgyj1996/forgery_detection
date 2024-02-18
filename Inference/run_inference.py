import os
import sys
import torch
import logging
import argparse
import numpy as np
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL


sys.path.append("../")
from Inference.forgery_pipeline import ForgeryEstimationPipeline
from utils.seed_all import seed_all
from utils.image_util import BayarConv2d


if __name__ == "__main__":
    use_seperate = True
    stable_diffusion_repo_path = "/data/huangjiaming/stable-diffusion-2"
    logging.basicConfig(level=logging.INFO)
    
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run Forgery Estimation using Stable Diffusion."
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default='None',
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--input_rgb_path",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of dataset.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, output mask at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--color_map",
        type=str,
        default="Spectral",
        help="Colormap used to render mask predictions.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Inference batch size. Default: 0 (will be set automatically).",
    )
    parser.add_argument(
        "--use_bayarConv",
        action="store_true",
        default=False,
        help="use bayarconv or not, scheme 2",
    )
    parser.add_argument(
        "--use_resnet50",
        action="store_true",
        default=False,
        help="use resnet50 of mvssnet nsb or not, scheme 2"
    )
    parser.add_argument(
        "--not_use_vae_rgb",
        action="store_true",
        default=False,
        help="not use vae rgb or use, scheme 1",
    )
    parser.add_argument(
        "--replace_vae_with_backbone",
        action="store_true",
        default=False,
        help="replace rgb vae with mvssnet backbone, scheme 3",
    )
    parser.add_argument(
        "--resnet50_weight",
        type=str,
        default='/data/huangjiaming/dataset/mvssnet_casia.pt',
        help="path to mvss weight"
    )
    
    args = parser.parse_args()
    checkpoint_path = args.pretrained_model_path
    input_image_path = args.input_rgb_path
    dataset_name = args.dataset_name
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")
    
    half_precision = args.half_precision

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res

    color_map = args.color_map
    seed = args.seed
    batch_size = args.batch_size
    
    if batch_size == 0:
        batch_size = 1
    
    # -------------------- Preparation --------------------
    # Random seed
    if seed is None:
        import time
        seed = int(time.time())
    seed_all(seed)

    # Output directories
    output_dir_color = os.path.join(output_dir, "colored")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------Data----------------------------
    logging.info("Inference Image Path from {}".format(input_image_path))

    # scheme 2: turn on use_bayarConv or use_resnet50, and close not_use_vae_rgb
    # scheme 1: turn on not_use_vae_rgb, and close use_bayarConv and use_resnet50
    in_channels = 8
    if args.use_resnet50:
        in_channels = 2056  # 2048 + 4 + 4
    elif args.use_bayarConv:
        in_channels = 11  # 3 + 4 + 4
    if args.not_use_vae_rgb:
        in_channels = 7  # 3 + 4, you can mod to 4 + 4 by concat max(features)
    if args.replace_vae_with_backbone:
        in_channels = 2052  # 2048 + 4, you can use conv 1x1 to reduce channel

    # -------------------- Model --------------------
    if half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32
    if not use_seperate:
        pipe = ForgeryEstimationPipeline.from_pretrained(checkpoint_path, torch_dtype=dtype)
        print("Using Completed")
    else:
        vae = AutoencoderKL.from_pretrained(os.path.join(stable_diffusion_repo_path, 'vae'))
        scheduler = DDIMScheduler.from_pretrained(os.path.join(stable_diffusion_repo_path, 'scheduler'))
        text_encoder = CLIPTextModel.from_pretrained(os.path.join(stable_diffusion_repo_path, 'text_encoder'))
        tokenizer = CLIPTokenizer.from_pretrained(os.path.join(stable_diffusion_repo_path, 'tokenizer'))
        unet = UNet2DConditionModel.from_pretrained(os.path.join(checkpoint_path, 'unet'),
                                                    in_channels=in_channels, sample_size=96,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)
        
        pipe = ForgeryEstimationPipeline(unet=unet,
                                         vae=vae,
                                         scheduler=scheduler,
                                         text_encoder=text_encoder,
                                         tokenizer=tokenizer,
                                         )
        print("Using Separated Modules")
    
    logging.info("loading pipeline whole successfully.")
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass  # run without xformers
    pipe = pipe.to(device)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for img_name in os.listdir(input_image_path):
            if not (img_name.endswith('.jpg') or img_name.endswith('.tif') or img_name.endswith('.png')):
                continue
            input_image_pil = Image.open(os.path.join(input_image_path, img_name))
            pipe_out = pipe(input_image_pil,
                            denoising_steps=denoise_steps,
                            ensemble_size=ensemble_size,
                            processing_res=processing_res,
                            match_input_res=match_input_res,
                            batch_size=batch_size,
                            color_map=color_map,
                            show_progress_bar=True,
                            use_bayarConv=args.use_bayarConv,
                            use_resnet50=args.use_resnet50,
                            not_use_vae_rgb=args.not_use_vae_rgb,
                            replace_vae_with_backbone=args.replace_vae_with_backbone,
                            resnet50_weight=args.resnet50_weight,
                            )

            forgery_pred = pipe_out.forgery_np
            forgery_colored = pipe_out.forgery_colored
            rgb_name_base = os.path.splitext(os.path.basename(img_name))[0]

            # Colorize
            colored_save_path = os.path.join(output_dir_color, f"{rgb_name_base}_colored.png")
            if os.path.exists(colored_save_path):
                logging.warning(f"Existing file: '{colored_save_path}' will be overwritten")

            # load test path of gt according to your setting
            if dataset_name == 'coco':
                mask = Image.open(os.path.join(input_image_path, img_name.split('.')[0] + '.png').replace('test_img', 'test_gt'))
            elif dataset_name == 'casia':
                mask = Image.open(os.path.join(input_image_path, img_name.split('.')[0] + '.png').replace('val_img','val_gt'))
            else:
                assert False, 'dataset name setting error %s' % dataset_name
            mask = np.array(mask)
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, 2)
            else:
                mask = mask[:, :, 0:1]
            mask = np.repeat(mask, 3, 2)
            forgery_colored = np.array(forgery_colored)
            input_img_np = np.array(input_image_pil)
            forgery_pred = np.expand_dims(forgery_pred, 2)
            forgery_pred = np.repeat(forgery_pred, 3, 2)
            forgery_pred = (forgery_pred * 255).astype(np.uint8)
            #np_tmp = np.concatenate((input_img_np, mask, forgery_pred, forgery_colored), axis=1)
            np_tmp = np.concatenate((forgery_colored), axis=1)
            forgery_colored = Image.fromarray(np_tmp)
            forgery_colored.save(colored_save_path)
