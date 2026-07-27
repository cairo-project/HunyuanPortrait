"""Installable library wrapper for HunyuanPortrait video-driven portrait animation.

HunyuanPortrait is VIDEO-DRIVEN: given a single source image and a driving
video, it animates the source portrait to follow the head pose and expression
of the driving video, producing an output video.

This module exposes a small, stable API on top of the original ``inference.py``
CLI:

    load_models(checkpoint_dir, device='cuda', dtype='fp16', **_) -> dict
    generate_video(models, source_image, driving_video, output_path, ...) -> str
    main()  # argparse CLI

The ``checkpoint_dir`` is the ``pretrained_weights`` directory described in the
project README, with the following layout::

    checkpoint_dir/
    ├── arcface.onnx
    ├── yoloface_v5m.pt
    ├── hyportrait/
    │   ├── dino.pth
    │   ├── expression.pth
    │   ├── headpose.pth
    │   ├── image_proj.pth
    │   ├── motion_proj.pth
    │   ├── pose_guider.pth
    │   └── unet.pth
    ├── scheduler/scheduler_config.json      # from stable-video-diffusion-img2vid-xt
    ├── unet/config.json                     # from stable-video-diffusion-img2vid-xt
    └── vae/                                  # from stable-video-diffusion-img2vid-xt
        ├── config.json
        └── diffusion_pytorch_model.fp16.safetensors
"""

import argparse
import os

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

from diffusers import AutoencoderKLTemporalDecoder

from src.dataset.test_preprocess import preprocess
from src.dataset.utils import (
    save_videos_grid,
    save_videos_from_pil,
    seed_everything,
    get_head_exp_motion_bucketid,
)
from src.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from src.pipelines.hunyuan_svd_pipeline import HunyuanLongSVDPipeline
from src.models.condition.unet_3d_svd_condition_ip import (
    UNet3DConditionSVDModel,
    init_ip_adapters,
)
from src.models.condition.coarse_motion import HeadExpression, HeadPose
from src.models.condition.refine_motion import IntensityAwareMotionRefiner
from src.models.condition.pose_guider import PoseGuider
from src.models.dinov2.models.vision_transformer import vit_large, ImageProjector

# Reuse the paste-back helpers from the original CLI so behaviour stays identical.
from inference import create_soft_mask, paste_back_frame

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_HERE, "config", "hunyuan-portrait.yaml")

_DTYPE_MAP = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPE_MAP[str(dtype).lower()]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype!r}")


def _build_config(checkpoint_dir, config_path=None):
    """Load the base config and point every checkpoint path at ``checkpoint_dir``.

    ``checkpoint_dir`` corresponds to the README's ``pretrained_weights`` folder.
    """
    cfg = OmegaConf.load(config_path or _DEFAULT_CONFIG)

    hy = os.path.join(checkpoint_dir, "hyportrait")
    cfg.pretrained_model_name_or_path = checkpoint_dir
    cfg.det_path = os.path.join(checkpoint_dir, "yoloface_v5m.pt")
    cfg.arcface_model_path = os.path.join(checkpoint_dir, "arcface.onnx")
    cfg.unet_checkpoint_path = os.path.join(hy, "unet.pth")
    cfg.pose_guider_checkpoint_path = os.path.join(hy, "pose_guider.pth")
    cfg.dino_checkpoint_path = os.path.join(hy, "dino.pth")
    cfg.image_proj_checkpoint_path = os.path.join(hy, "image_proj.pth")
    cfg.motion_expression_checkpoint_path = os.path.join(hy, "expression.pth")
    cfg.motion_pose_checkpoint_path = os.path.join(hy, "headpose.pth")
    cfg.motion_proj_checkpoint_path = os.path.join(hy, "motion_proj.pth")
    return cfg


@torch.no_grad()
def load_models(checkpoint_dir, device="cuda", dtype="fp16", config_path=None, **_):
    """Load all HunyuanPortrait models and assemble the inference pipeline.

    Args:
        checkpoint_dir: Path to the ``pretrained_weights`` directory (see README).
        device: Torch device string, e.g. ``"cuda"`` or ``"cpu"``.
        dtype: Weight dtype: ``"fp16"``, ``"fp32"``, ``"bf16"`` or a torch.dtype.
        config_path: Optional override for the YAML config
            (defaults to ``config/hunyuan-portrait.yaml``).

    Returns:
        dict containing the assembled ``pipeline`` plus the individual ``models``
        and configuration needed by :func:`generate_video`.
    """
    cfg = _build_config(checkpoint_dir, config_path)
    weight_dtype = _resolve_dtype(dtype)
    cfg.weight_dtype = {
        torch.float16: "fp16",
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
    }[weight_dtype]

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="vae", variant="fp16"
    )
    scheduler = EulerDiscreteScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path, subfolder="scheduler"
    )
    unet = UNet3DConditionSVDModel.from_config(
        cfg.pretrained_model_name_or_path, subfolder="unet", variant="fp16"
    )
    init_ip_adapters(unet, cfg.num_adapter_embeds, cfg.ip_motion_scale)

    pose_guider = PoseGuider(
        conditioning_embedding_channels=320,
        block_out_channels=(16, 32, 96, 256),
    ).to(device=device)

    motion_expression_model = HeadExpression(cfg.input_expression_dim).to(device)
    motion_headpose_model = HeadPose().to(device)
    motion_proj = IntensityAwareMotionRefiner(
        input_dim=cfg.input_expression_dim,
        output_dim=cfg.motion_expression_dim,
        num_queries=cfg.num_queries,
    ).to(device=device)

    image_encoder = vit_large(
        patch_size=14,
        num_register_tokens=4,
        img_size=526,
        init_values=1.0,
        block_chunks=0,
        backbone=True,
        layers_output=True,
        add_adapter_layer=[3, 7, 11, 15, 19, 23],
        visual_adapter_dim=384,
    )
    image_proj = ImageProjector(
        cfg.num_img_tokens, cfg.num_queries, dtype=unet.dtype
    ).to(device=device)

    # Load checkpoints.
    image_encoder.load_state_dict(torch.load(cfg.dino_checkpoint_path), strict=True)
    image_proj.load_weights(cfg.image_proj_checkpoint_path, strict=True)
    pose_guider.load_state_dict(
        torch.load(cfg.pose_guider_checkpoint_path, map_location="cpu"), strict=True
    )
    unet.load_state_dict(
        torch.load(cfg.unet_checkpoint_path, map_location="cpu"), strict=True
    )
    motion_proj.load_state_dict(
        torch.load(cfg.motion_proj_checkpoint_path, map_location="cpu"), strict=True
    )
    motion_expression_model.load_state_dict(
        torch.load(cfg.motion_expression_checkpoint_path, map_location=device),
        strict=True,
    )
    motion_headpose_model.load_state_dict(
        torch.load(cfg.motion_pose_checkpoint_path, map_location=device), strict=True
    )

    for m in (
        image_encoder,
        image_proj,
        pose_guider,
        unet,
        motion_proj,
        motion_expression_model,
        motion_headpose_model,
    ):
        m.eval()
    motion_expression_model.requires_grad_(False)
    motion_headpose_model.requires_grad_(False)

    vae.to(weight_dtype)
    unet.to(weight_dtype)
    pose_guider.to(weight_dtype)
    image_encoder.to(weight_dtype)
    image_proj.to(weight_dtype)

    pipeline = HunyuanLongSVDPipeline(
        unet=unet,
        image_encoder=image_encoder,
        image_proj=image_proj,
        vae=vae,
        pose_guider=pose_guider,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device, dtype=unet.dtype)

    arcface_session = None
    if cfg.use_arcface:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider"]
            if str(device).startswith("cuda")
            else ["CPUExecutionProvider"]
        )
        arcface_session = ort.InferenceSession(
            cfg.arcface_model_path, providers=providers
        )

    return {
        "pipeline": pipeline,
        "cfg": cfg,
        "device": device,
        "weight_dtype": weight_dtype,
        "arcface_session": arcface_session,
        "models": {
            "vae": vae,
            "unet": unet,
            "scheduler": scheduler,
            "pose_guider": pose_guider,
            "image_encoder": image_encoder,
            "image_proj": image_proj,
            "motion_proj": motion_proj,
            "motion_expression_model": motion_expression_model,
            "motion_headpose_model": motion_headpose_model,
        },
    }


@torch.no_grad()
def generate_video(
    models,
    source_image,
    driving_video,
    output_path,
    seed=42,
    num_frames=49,
    fps=25,
    **_,
):
    """Animate ``source_image`` following ``driving_video`` and write a video.

    Args:
        models: The dict returned by :func:`load_models`.
        source_image: Path to the source portrait image.
        driving_video: Path to the driving video.
        output_path: Path to write the full-resolution (paste-back) result video.
        seed: Random seed.
        num_frames: Maximum number of driving frames to consume.
        fps: Output video frame rate.

    Returns:
        The ``output_path`` string.
    """
    cfg = models["cfg"]
    device = models["device"]
    pipe = models["pipeline"]
    m = models["models"]
    arcface_session = models["arcface_session"]

    motion_expression_model = m["motion_expression_model"]
    motion_headpose_model = m["motion_headpose_model"]
    motion_proj = m["motion_proj"]

    if seed is not None:
        seed_everything(seed)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sample = preprocess(
        source_image,
        driving_video,
        limit=num_frames if num_frames is not None else cfg.frame_num,
        image_size=cfg.arcface_img_size,
        area=cfg.area,
        det_path=cfg.det_path,
    )

    original_image = sample["original_image"]
    crop_bbox = sample["crop_bbox"]
    ref_img = sample["ref_img"].unsqueeze(0).to(device)
    transformed_images = sample["transformed_images"].unsqueeze(0).to(device)
    arcface_img = sample["arcface_image"]
    lmk_list = sample["lmk_list"]

    if not cfg.use_arcface or arcface_img is None or arcface_session is None:
        arcface_embeddings = np.zeros((1, cfg.arcface_img_size))
    else:
        arcface_img = arcface_img.transpose((2, 0, 1)).astype(np.float32)[np.newaxis, ...]
        arcface_embeddings = arcface_session.run(None, {"data": arcface_img})[0]
        arcface_embeddings = arcface_embeddings / np.linalg.norm(arcface_embeddings)

    dwpose_images = sample["img_pose"]
    motion_pose_images = sample["motion_pose_image"]
    motion_face_images = sample["motion_face_image"]
    driven_images = sample["driven_image"]

    pose_cond_tensor_all = []
    driven_feat_all = []
    uncond_driven_feat_all = []
    num_frames_all = 0
    driven_video_all = []
    batch = cfg.n_sample_frames

    for idx in range(0, motion_pose_images.shape[0], batch):
        driven_video = driven_images[idx : idx + batch].to(device)
        motion_pose_image = motion_pose_images[idx : idx + batch].to(device)
        motion_face_image = motion_face_images[idx : idx + batch].to(device)
        pose_cond_tensor = dwpose_images[idx : idx + batch].to(device)
        lmks = lmk_list[idx : idx + batch]
        n = motion_pose_image.shape[0]

        motion_bucket_id_head, motion_bucket_id_exp = get_head_exp_motion_bucketid(lmks)

        motion_feature = motion_expression_model(motion_face_image)
        motion_bucket_id_head = torch.IntTensor([motion_bucket_id_head]).to(device)
        motion_bucket_id_exp = torch.IntTensor([motion_bucket_id_exp]).to(device)
        motion_feature_embed = motion_proj(
            motion_feature, motion_bucket_id_head, motion_bucket_id_exp
        )

        driven_pose_feat = motion_headpose_model(motion_pose_image * 2 + 1)
        driven_pose_feat_embed = torch.cat(
            [driven_pose_feat["rotation"], driven_pose_feat["translation"] * 0], dim=-1
        )

        driven_feat = torch.cat(
            [
                motion_feature_embed,
                driven_pose_feat_embed.unsqueeze(1).repeat(
                    1, motion_feature_embed.shape[1], 1
                ),
            ],
            dim=-1,
        )
        driven_feat = driven_feat.unsqueeze(0)
        uncond_driven_feat = torch.zeros_like(driven_feat)

        pose_cond_tensor = pose_cond_tensor.unsqueeze(0)
        pose_cond_tensor = rearrange(pose_cond_tensor, "b f c h w -> b c f h w")

        pose_cond_tensor_all.append(pose_cond_tensor)
        driven_feat_all.append(driven_feat)
        uncond_driven_feat_all.append(uncond_driven_feat)
        driven_video_all.append(driven_video)
        num_frames_all += n

    driven_video_all = torch.cat(driven_video_all, dim=0)
    pose_cond_tensor_all = torch.cat(pose_cond_tensor_all, dim=2)
    uncond_driven_feat_all = torch.cat(uncond_driven_feat_all, dim=1)
    driven_feat_all = torch.cat(driven_feat_all, dim=1)

    # Pad head/tail with a smooth ramp so motion eases in and out.
    driven_video_all_2 = []
    pose_cond_tensor_all_2 = []
    driven_feat_all_2 = []
    uncond_driven_feat_all_2 = []

    for i in range(cfg.pad_frames):
        weight = i / cfg.pad_frames
        driven_video_all_2.append(driven_video_all[:1])
        pose_cond_tensor_all_2.append(pose_cond_tensor_all[:, :, :1])
        driven_feat_all_2.append(driven_feat_all[:, :1] * weight)
        uncond_driven_feat_all_2.append(uncond_driven_feat_all[:, :1])

    driven_video_all_2.append(driven_video_all)
    pose_cond_tensor_all_2.append(pose_cond_tensor_all)
    driven_feat_all_2.append(driven_feat_all)
    uncond_driven_feat_all_2.append(uncond_driven_feat_all)

    for i in range(cfg.pad_frames):
        weight = i / cfg.pad_frames
        driven_video_all_2.append(driven_video_all[:1])
        pose_cond_tensor_all_2.append(pose_cond_tensor_all[:, :, :1])
        driven_feat_all_2.append(driven_feat_all[:, -1:] * (1 - weight))
        uncond_driven_feat_all_2.append(uncond_driven_feat_all[:, :1])

    driven_video_all = torch.cat(driven_video_all_2, dim=0)
    pose_cond_tensor_all = torch.cat(pose_cond_tensor_all_2, dim=2)
    driven_feat_all = torch.cat(driven_feat_all_2, dim=1)
    uncond_driven_feat_all = torch.cat(uncond_driven_feat_all_2, dim=1)
    num_frames_all += cfg.pad_frames * 2

    video = pipe(
        ref_img.clone(),
        transformed_images.clone(),
        pose_cond_tensor_all,
        driven_feat_all,
        uncond_driven_feat_all,
        height=cfg.height,
        width=cfg.width,
        num_frames=num_frames_all,
        decode_chunk_size=cfg.decode_chunk_size,
        motion_bucket_id=cfg.motion_bucket_id,
        fps=cfg.fps,
        noise_aug_strength=cfg.noise_aug_strength,
        min_guidance_scale1=cfg.min_appearance_guidance_scale,
        max_guidance_scale1=cfg.max_appearance_guidance_scale,
        min_guidance_scale2=cfg.min_motion_guidance_scale,
        max_guidance_scale2=cfg.max_motion_guidance_scale,
        overlap=cfg.overlap,
        shift_offset=cfg.shift_offset,
        frames_per_batch=cfg.n_sample_frames,
        num_inference_steps=cfg.num_inference_steps,
        i2i_noise_strength=cfg.i2i_noise_strength,
        arcface_embeddings=arcface_embeddings,
    ).frames

    video = (video * 0.5 + 0.5).clamp(0, 1).cpu()
    if cfg.pad_frames > 0:
        video = video[:, :, cfg.pad_frames : -cfg.pad_frames]

    # Paste the generated crop back into the original full-resolution frame.
    from PIL import Image

    generated_frames = video[0].permute(1, 2, 3, 0).numpy()
    generated_frames = (generated_frames * 255).astype(np.uint8)

    x1, y1, x2, y2 = crop_bbox
    soft_mask = create_soft_mask((x2 - x1, y2 - y1))

    final_frames = [
        Image.fromarray(paste_back_frame(original_image, gen_frame, crop_bbox, soft_mask))
        for gen_frame in generated_frames
    ]

    save_videos_from_pil(final_frames, output_path, fps=fps)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="HunyuanPortrait video-driven portrait animation (library CLI)."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="pretrained_weights",
        help="Path to the pretrained_weights directory (see README).",
    )
    parser.add_argument("--source_image", "--image_path", dest="source_image",
                        type=str, required=True, help="Source portrait image.")
    parser.add_argument("--driving_video", "--video_path", dest="driving_video",
                        type=str, required=True, help="Driving video.")
    parser.add_argument("--output_path", type=str, default="output.mp4",
                        help="Output video path.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config override.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "fp32", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    models = load_models(
        args.checkpoint_dir,
        device=args.device,
        dtype=args.dtype,
        config_path=args.config,
    )
    out = generate_video(
        models,
        args.source_image,
        args.driving_video,
        args.output_path,
        seed=args.seed,
        num_frames=args.num_frames,
        fps=args.fps,
    )
    print(f"Saved output video to {out}")


if __name__ == "__main__":
    main()
