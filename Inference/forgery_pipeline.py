from typing import Any, Dict, Union

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import torch.nn.functional as F
from diffusers import DiffusionPipeline, DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer

from utils.image_util import resize_max_res, chw2hwc, colorize_forgery_maps, BayarConv2d, rgb2gray, ResNet50
from utils.ensemble import ensemble_masks


class ForgeryPipelineOutput(BaseOutput):
    """
    Output class for forgery pipeline.
    Args:
        forgery_np (`np.ndarray`):
            Predicted forgery map, with values in the range of [0, 1].
        forgery_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    forgery_np: np.ndarray
    forgery_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class ForgeryEstimationPipeline(DiffusionPipeline):
    rgb_latent_scale_factor = 0.18215
    forgery_latent_scale_factor = 0.18215

    def __init__(self,
                 unet: UNet2DConditionModel,
                 vae: AutoencoderKL,
                 scheduler: DDIMScheduler,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )
        self.empty_text_embed = None


    @torch.no_grad()
    def __call__(self,
                 input_image: Image,
                 denoising_steps: int = 10,
                 ensemble_size: int = 10,
                 processing_res: int = 768,
                 match_input_res: bool = True,
                 batch_size: int = 0,
                 color_map: str = "Spectral",
                 show_progress_bar: bool = True,
                 use_bayarConv: bool = False,
                 use_resnet50: bool = False,
                 not_use_vae_rgb: bool = False,
                 replace_vae_with_backbone: bool = False,
                 resnet50_weight: str = '/data/huangjiaming/dataset/mvssnet_casia.pt',
                 ensemble_kwargs: Dict = None,
                 ) -> ForgeryPipelineOutput:

        # inherit from thea Diffusion Pipeline
        device = self.device
        input_size = input_image.size
        self.use_bayarConv = use_bayarConv
        self.use_resnet50 = use_resnet50
        self.not_use_vae_rgb = not_use_vae_rgb
        self.replace_vae_with_backbone = replace_vae_with_backbone
        pretrained_dict = torch.load(resnet50_weight)
        self.bayarConv = BayarConv2d(in_channels=1, out_channels=3, padding=2)
        pretrained_bayarConv = {'kernel': pretrained_dict['constrain_conv.kernel']}
        self.bayarConv.load_state_dict(pretrained_bayarConv, strict=True)

        self.bayar_extractor = ResNet50(n_input=3)
        pretrained_extractor = {k[16:]: v for k, v in pretrained_dict.items() if 'noise_extractor.' in k}
        self.bayar_extractor.load_state_dict(pretrained_extractor, strict=True)

        self.bayar_backbone = ResNet50(n_input=3)
        self.bayar_backbone.load_state_dict(pretrained_dict, strict=False)

        self.bayarConv.to(self.device)
        self.bayar_extractor.to(self.device)
        self.bayar_backbone.to(self.device)

        # adjust the input resolution.
        if not match_input_res:
            assert (
                    processing_res is not None
            ), " Value Error: `resize_output_back` is only valid with "

        assert processing_res >= 0
        assert denoising_steps >= 1
        assert ensemble_size >= 1

        # --------------- Image Processing ------------------------
        # Resize image
        if processing_res > 0:
            input_image = resize_max_res(
                input_image, max_edge_resolution=processing_res
            )
        # Convert the image to RGB
        input_image = input_image.convert("RGB")
        image = np.array(input_image)
        # Normalize RGB Values
        rgb = np.transpose(image, (2, 0, 1))  # [H, W, c] -> [c, H, W]
        rgb_norm = rgb / 255.0
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)
        assert rgb_norm.min() >= 0.0 and rgb_norm.max() <= 1.0

        # ----------------- predicting forgery -----------------
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)

        # find the batch size
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = 1
        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)

        # predict the forgery
        forgery_pred_ls = []
        if show_progress_bar:
            iterable_bar = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable_bar = single_rgb_loader

        for batch in iterable_bar:
            (batched_image,) = batch
            forgery_pred_raw = self.single_infer(
                rgb_in=batched_image,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
            )
            forgery_pred_ls.append(forgery_pred_raw.detach().clone())

        forgery_preds = torch.concat(forgery_pred_ls, axis=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            forgery_pred, forgery_uncert = ensemble_masks(forgery_preds, **(ensemble_kwargs or {}))
        else:
            forgery_pred = forgery_preds
            forgery_uncert = None

        # ----------------- Post processing -----------------
        # Scale prediction to [0, 1]
        min_d = torch.min(forgery_pred)
        max_d = torch.max(forgery_pred)
        forgery_pred = (forgery_pred - min_d) / (max_d - min_d)

        # Convert to numpy
        forgery_pred = forgery_pred.cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        if match_input_res:
            pred_img = Image.fromarray(forgery_pred)
            pred_img = pred_img.resize(input_size)
            forgery_pred = np.asarray(pred_img)

        # Clip output range: current size is the original size
        forgery_pred = forgery_pred.clip(0, 1)

        # Colorize
        forgery_colored = colorize_forgery_maps(forgery_pred, 0, 1, cmap=color_map).squeeze()
        forgery_colored = (forgery_colored * 255).astype(np.uint8)
        forgery_colored_hwc = chw2hwc(forgery_colored)
        forgery_colored_img = Image.fromarray(forgery_colored_hwc)

        return ForgeryPipelineOutput(
            forgery_np=forgery_pred,
            forgery_colored=forgery_colored_img,
            uncertainty=forgery_uncert)

    def __encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(self, rgb_in: torch.Tensor, num_inference_steps: int, show_pbar: bool):
        device = rgb_in.device
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # encode bayar
        bayar_latent = self.encode_bayar(rgb_in)

        # encode backbone
        backbone_latent = self.encode_backbone(rgb_in)

        # Initial forgery (noise)
        forgery_latent = torch.randn(rgb_latent.shape, device=device, dtype=self.dtype)

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat((rgb_latent.shape[0], 1, 1))  # B, 2, 1024

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            if self.not_use_vae_rgb:
                unet_input = torch.cat([bayar_latent, forgery_latent], dim=1)
            elif self.replace_vae_with_backbone:
                unet_input = torch.cat([backbone_latent, forgery_latent], dim=1)
            elif self.use_bayarConv:
                unet_input = torch.cat([rgb_latent, bayar_latent, forgery_latent], dim=1)
            else:
                unet_input = torch.cat([rgb_latent, forgery_latent], dim=1)
            noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            forgery_latent = self.scheduler.step(noise_pred, t, forgery_latent).prev_sample

        torch.cuda.empty_cache()
        forgery = self.decode_forgery(forgery_latent)
        # clip prediction
        forgery = torch.clip(forgery, -1.0, 1.0)
        # shift to [0, 1]
        forgery = (forgery + 1.0) / 2.0

        return forgery

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.
        """
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.rgb_latent_scale_factor

        return rgb_latent

    def encode_bayar(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.
        """
        bayar = rgb2gray(rgb_in)
        bayar = self.bayarConv(bayar)
        bayar = F.interpolate(bayar, scale_factor=0.5, mode='bilinear')
        bayar = F.interpolate(bayar, scale_factor=0.5, mode='bilinear')
        bayar_latent = F.interpolate(bayar, scale_factor=0.5, mode='bilinear')

        if self.use_bayarConv:
            bayar_latent = rgb2gray(rgb_in)
            bayar_latent = self.bayarConv(bayar_latent)
            if self.use_resnet50:
                constrain_features, _ = self.bayar_extractor.base_forward(bayar_latent)
                bayar_latent = constrain_features[-1]
                bayar_latent = F.interpolate(bayar_latent, scale_factor=2, mode='bilinear')
            else:
                bayar_latent = F.interpolate(bayar_latent, scale_factor=0.5, mode='bilinear')
                bayar_latent = F.interpolate(bayar_latent, scale_factor=0.5, mode='bilinear')
                bayar_latent = F.interpolate(bayar_latent, scale_factor=0.5, mode='bilinear')

        if self.not_use_vae_rgb:
            bayar_latent = rgb2gray(rgb_in)
            bayar_latent = self.bayarConv(bayar_latent)
            bayar_latent = F.interpolate(bayar_latent, scale_factor=0.5, mode='bilinear')
            bayar_latent = F.interpolate(bayar_latent, scale_factor=0.5, mode='bilinear')
            bayar_latent = F.interpolate(bayar_latent, scale_factor=0.5, mode='bilinear')

        return bayar_latent

    def encode_backbone(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.
        """
        backbones, _ = self.bayar_backbone.base_forward(rgb_in)
        backbone_latent = backbones[-1]
        backbone_latent = F.interpolate(backbone_latent, scale_factor=2, mode='bilinear')

        return backbone_latent

    def decode_forgery(self, forgery_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent into forgery map.
        """
        forgery_latent = forgery_latent / self.forgery_latent_scale_factor
        z = self.vae.post_quant_conv(forgery_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        forgery_mean = stacked.mean(dim=1, keepdim=True)
        return forgery_mean
