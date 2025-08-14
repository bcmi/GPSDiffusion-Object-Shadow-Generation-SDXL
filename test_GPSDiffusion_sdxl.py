#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import itertools
import argparse
import functools
import gc
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
import types
import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from torch.utils.data import default_collate
import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from pytorch_lightning import seed_everything
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import cv2
import pickle
import numpy as np
from torch.utils.data import Dataset
from skimage import color
from PIL import Image
import os
from base_network import MaskCls, RegNetwork

seed_everything(42)

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

if is_torch2_available():
    from attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, XFormersAttnProcessor
else:
    from attention_processor import IPAttnProcessor, AttnProcessor, AttnProcessor2_0, XFormersAttnProcessor


def print_memory(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[Memory] {prefix} Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

class TestDataset(Dataset):
    def __init__(self, data_file_path, text_encoders, tokenizers, caption_column='text'):
        self.data_root = data_file_path
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.caption_column = caption_column

        with open(os.path.join(self.data_root, 'TestLabel.txt'), 'r') as f:
            self.data = [line.strip() for line in f.readlines()]

        self.prompts = ['foreground object with shadow'] * len(self.data)

        # Try load from cache
        self.load_or_generate_embeddings()

    def embedding_cache_paths(self):
        return {
            'prompt_embeds': os.path.join(self.data_root, 'prompt_embeds.pt'),
            'text_embeds': os.path.join(self.data_root, 'text_embeds.pt'),
            'time_ids': os.path.join(self.data_root, 'time_ids.pt')
        }

    def load_or_generate_embeddings(self):
        cache = self.embedding_cache_paths()
        if all(os.path.exists(p) for p in cache.values()):
            print(" Loading prompt embeddings from cache...")
            self.prompt_embeds = torch.load(cache['prompt_embeds'])
            self.text_embeds = torch.load(cache['text_embeds'])
            self.time_ids = torch.load(cache['time_ids'])
        else:
            print(" Computing prompt embeddings and saving cache...")
            self.compute_and_save_embeddings(cache)

    def compute_and_save_embeddings(self, cache, batch_size=64):
        prompt_embeds, text_embeds, time_ids = [], [], []
        for i in tqdm(range(0, len(self.prompts), batch_size), desc="Embedding"):
            batch_prompts = self.prompts[i:i + batch_size]
            dummy_batch = {
                self.caption_column: batch_prompts,
                "__fake_image__": [None] * len(batch_prompts)
            }

            with torch.no_grad():
                embeddings = compute_embeddings(
                    batch=dummy_batch,
                    proportion_empty_prompts=args.proportion_empty_prompts,
                    text_encoders=self.text_encoders,
                    tokenizers=self.tokenizers,
                    is_train=False
                )

            prompt_embeds.append(embeddings["prompt_embeds"].cpu())
            text_embeds.append(embeddings["text_embeds"].cpu())
            time_ids.append(embeddings["time_ids"].cpu())

        self.prompt_embeds = torch.cat(prompt_embeds, dim=0)
        self.text_embeds = torch.cat(text_embeds, dim=0)
        self.time_ids = torch.cat(time_ids, dim=0)

        torch.save(self.prompt_embeds, cache['prompt_embeds'])
        torch.save(self.text_embeds, cache['text_embeds'])
        torch.save(self.time_ids, cache['time_ids'])

    def __len__(self):
        return len(self.data)

    def load_image(self, folder, name):
        img = cv2.imread(os.path.join(self.data_root, folder, name))
        img = cv2.cvtColor(cv2.resize(img, (args.resolution, args.resolution)), cv2.COLOR_BGR2RGB)
        return img

    def load_mask(self, folder, name):
        mask = cv2.imread(os.path.join(self.data_root, folder, name), cv2.IMREAD_GRAYSCALE)
        return cv2.resize(mask, (args.resolution, args.resolution))

    def compute_bbx(self, mask):
        _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return torch.zeros((5,), dtype=torch.int32), torch.zeros((args.resolution, args.resolution), dtype=torch.float16)
        merged = np.concatenate(contours)
        (x, y), (w, h), theta = cv2.minAreaRect(merged)
        if w < h:
            w, h = h, w
            theta += 90
        bbx = np.array([x, y, w + 1, h + 1, theta]).astype(int)
        region = torch.zeros((args.resolution, args.resolution), dtype=torch.float16)
        return torch.tensor(bbx), region

    def __getitem__(self, idx):
        pic_name = self.data[idx]

        shadowfree_img = self.load_image('shadowfree_imgs', pic_name)
        object_mask = self.load_mask('object_masks', pic_name)
        shadow_img = self.load_image('shadowfree_imgs', pic_name)

        control_input = np.concatenate([shadowfree_img, object_mask[..., None]], axis=-1)
        control_input = control_input.astype(np.float32) / 255.0
        target = (shadow_img.astype(np.float32) / 127.5) - 1.0

        bbx_tensor, bbx_region = self.compute_bbx(object_mask)

        return {
            "pixel_values": torch.tensor(target).permute(2, 0, 1).float(),
            "conditioning_pixel_values": torch.tensor(control_input).permute(2, 0, 1).float(),
            "bbx": bbx_region,
            "fg": bbx_tensor,
            "embeddings": torch.zeros((64, 2048), dtype=torch.float16),
            "prompt_ids": self.prompt_embeds[idx],
            "unet_added_conditions": {
                "text_embeds": self.text_embeds[idx],
                "time_ids": self.time_ids[idx]
            }
        }

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs, down_block_additional_residuals, mid_block_additional_residual, mask_embeddings):
        ip_tokens = self.image_proj_model(mask_embeddings)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(sample=noisy_latents, timestep=timesteps, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs, down_block_additional_residuals=down_block_additional_residuals, mid_block_additional_residual=mid_block_additional_residual, return_dict=False)[0]
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=768, extra_embeddings_dim=2048, clip_extra_context_tokens=64):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(extra_embeddings_dim, cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        extra_embeds = self.proj(image_embeds)
        extra_embeds = self.norm(extra_embeds)
        return extra_embeds
    
class ProcessedDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def compute_embeddings(batch, proportion_empty_prompts, text_encoders, tokenizers, is_train=True):
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (args.crops_coords_top_left_h, args.crops_coords_top_left_w)
        prompt_batch = batch[args.caption_column]
 
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train
        )
        add_text_embeds = pooled_prompt_embeds

        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])

        prompt_embeds = prompt_embeds
        add_text_embeds = add_text_embeds
        add_time_ids = add_time_ids.repeat(len(prompt_batch), 1)
        add_time_ids = add_time_ids.to(prompt_embeds.dtype)
        unet_added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

        return {"prompt_embeds": prompt_embeds, **unet_added_cond_kwargs}

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)
if is_torch_npu_available():
    torch.npu.config.allow_internal_format = False


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument('--dataset_path', default='./data/desobav2', type=str)
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to an improved VAE to stabilize training. For more details check out: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default='./pretrained_models/controlnet',
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ckpts_GPS_sdxl",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default="./pretrained_models/ip_adapter.ckpt",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--test_result_dir",
        type=str,
        default="./results_sdxl/gen_result",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--crops_coords_top_left_h",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--crops_coords_top_left_w",
        type=int,
        default=0,
        help=("Coordinate for (the height) to be included in the crop coordinate embeddings needed by SDXL UNet."),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--enable_npu_flash_attention", action="store_true", help="Whether or not to use npu flash attention."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="sd_xl_train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")


    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoders, tokenizers, proportion_empty_prompts, is_train=True):
    prompt_embeds_list = []

    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            bs_embed, seq_len, _ = prompt_embeds.shape
            prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
            prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def collate_fn(examples):
    pixel_values = torch.stack([torch.from_numpy(example["pixel_values"].transpose(2, 0, 1)) if isinstance(example["pixel_values"], np.ndarray) else example["pixel_values"].permute(0, 3, 1, 2) for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([torch.from_numpy(example["conditioning_pixel_values"].transpose(2, 0, 1)) if isinstance(example["conditioning_pixel_values"], np.ndarray) else example["conditioning_pixel_values"].permute(0, 3, 1, 2) for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompt_ids = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])

    add_text_embeds = torch.stack([torch.tensor(example["text_embeds"]) for example in examples])
    add_time_ids = torch.stack([torch.tensor(example["time_ids"]) for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_ids": prompt_ids,
        "unet_added_conditions": {"text_embeds": add_text_embeds, "time_ids": add_time_ids},
    }


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    mask_cls = MaskCls(num_classes=256)
    if os.path.exists('./pretrained_models/Shadow_cls.pth'):
        mask_cls.load_state_dict(torch.load('./pretrained_models/Shadow_cls.pth')['net'])
        print('Loading mask embeddings classification model successfully')
    bbx_reg = RegNetwork()
    if os.path.exists('./pretrained_models/Shadow_reg.pth'):
        bbx_reg.load_state_dict(torch.load('./pretrained_models/Shadow_reg.pth')['net'])
        print('Loading rotated bounding box regression model successfully')

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet, conditioning_channels=5)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)
        
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    mask_cls.requires_grad_(False)
    bbx_reg.requires_grad_(False)
    controlnet.requires_grad_(False)

    #ip-adapter
    num_tokens = 4
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        extra_embeddings_dim=2048,
        clip_extra_context_tokens=64,
    ).to(accelerator.device)

    # init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    ip_adapter.requires_grad_(False)
    
    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            logger.info("npu flash attention enabled.")
            unet.enable_npu_flash_attention()
        else:
            raise ValueError("npu flash attention requires torch_npu extensions and is supported only on npu devices.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    # The VAE is in float32 to avoid NaN losses.
    if args.pretrained_vae_model_name_or_path is not None:
        vae.to(accelerator.device, dtype=weight_dtype)
    else:
        vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    mask_cls.to(accelerator.device, dtype=weight_dtype)
    bbx_reg.to(accelerator.device, dtype=weight_dtype)
    
    # Let's first compute all the embeddings so that we can free up the text encoders
    # from memory.
    text_encoders = [text_encoder_one, text_encoder_two]
    tokenizers = [tokenizer_one, tokenizer_two]

    # train_dataset = get_train_dataset(args, accelerator)
    test_dataset = TestDataset(
        data_file_path=args.dataset_path,
        text_encoders=[text_encoder_one, text_encoder_two],
        tokenizers=[tokenizer_one, tokenizer_two],
        caption_column='text',
    )

    del text_encoders, tokenizers
    gc.collect()
    torch.cuda.empty_cache()

    def custom_collate_fn(batch):
        batch = default_collate(batch)
        device = accelerator.device

        batch["prompt_ids"] = batch["prompt_ids"].to(device)

        def move_to_device(obj):
            if isinstance(obj, dict):
                return {k: move_to_device(v) for k, v in obj.items()}
            elif isinstance(obj, torch.Tensor):
                return obj.to(device)
            else:
                return obj

        batch["unet_added_conditions"] = move_to_device(batch["unet_added_conditions"])
        return batch

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=custom_collate_fn,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    controlnet, ip_adapter, test_dataloader = accelerator.prepare(
        controlnet, ip_adapter, test_dataloader
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    logger.info("***** Running testing *****")
    logger.info(f"  Num examples = {len(test_dataset)}")
    logger.info(f"  Num batches each epoch = {len(test_dataloader)}")

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing", unit="batch"):
            batch_name = batch["name"]  # List of original image names (no extension)
    
            for repeat in range(5):  # generate 5 times per image
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                latents = latents.to(weight_dtype)

                noise = torch.randn_like(latents)
 
                bsz = latents.shape[0]
                timesteps = torch.full((bsz,), 999, device=latents.device, dtype=torch.long)
                # timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents.float(), noise.float(), timesteps).to(device=latents.device, dtype=weight_dtype)
                latents = noisy_latents

                def gen_bbx(pred_t, fg_instance_bbx, mask):
                    img_height, img_width = 512, 512
                    pred_bbx = pred_t.new(pred_t.shape)
                    pred_bbx[:,0] = pred_t[:,0] * fg_instance_bbx[:,2] + fg_instance_bbx[:,0]
                    pred_bbx[:,1] = pred_t[:,1] * fg_instance_bbx[:,3] + fg_instance_bbx[:,1]
                    pred_bbx[:,2] = fg_instance_bbx[:,2] * torch.exp(pred_t[:,2])
                    pred_bbx[:,3] = fg_instance_bbx[:,3] * torch.exp(pred_t[:,3])
                    pred_bbx[:,4] = pred_t[:,4] * 180 / np.pi + fg_instance_bbx[:,4]

                    mask = mask.cpu().numpy()
                    bs = mask.shape[0]
                    for i in range(bs):
                        if pred_bbx[i,4] > 0:
                            temp = pred_bbx[i,2]
                            pred_bbx[i,2] = pred_bbx[i,3]
                            pred_bbx[i,3] = temp
                            pred_bbx[i,4] = pred_bbx[i,4] - 90
                        x, y, w, h, theta = pred_bbx[i,0], pred_bbx[i,1], pred_bbx[i,2], pred_bbx[i,3], pred_bbx[i,4]
                        box = ((x, y), (w, h), theta)
                        box_points = cv2.boxPoints(box)
                        perturbation = np.random.uniform(-5, 5, box_points.shape)
                        box_points = box_points + perturbation
                        box_points = np.clip(box_points, 0, [img_width - 1, img_height - 1])
                        box_points_int = np.int32(box_points)
                        cv2.fillPoly(mask[i].astype(np.uint8), [box_points_int], 1)
                    return torch.tensor(mask).unsqueeze(1)

                geometry_input = batch["conditioning_pixel_values"].to(dtype=weight_dtype, device=accelerator.device)
                pred_t = bbx_reg(geometry_input)
                bbx_mask = batch['bbx'].to(accelerator.device)
                fg_instance_bbx = batch['fg'].to(accelerator.device)
                bbx_region = gen_bbx(pred_t, fg_instance_bbx, bbx_mask).to(dtype=weight_dtype, device=accelerator.device)

                controlnet_image = torch.cat((geometry_input, bbx_region), 1)

                # Set scheduler timesteps (ensure descending order for most schedulers)
                num_inference_steps = 50
                noise_scheduler.set_timesteps(num_inference_steps)
                noise_scheduler.timesteps = noise_scheduler.timesteps.to(latents.device)  # ✅ overwrite in-place
                timesteps = noise_scheduler.timesteps

                # Denoising loop
                for t in tqdm(timesteps, desc=f"Denoising {batch_name[0]}_{repeat}", leave=False):
                    # Run ControlNet
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        latents,
                        t,
                        encoder_hidden_states=batch["prompt_ids"],
                        added_cond_kwargs=batch["unet_added_conditions"],
                        controlnet_cond=controlnet_image,
                        return_dict=False,
                    )

                    mask_label = mask_cls(geometry_input)
                    with open('./pretrained_models/Shadow_cls_label.pkl', 'rb') as f:
                        centroid_dict = pickle.load(f)
                    _, top64_label = torch.topk(mask_label, 64, largest=True, sorted=False)
                    mask_embeddings = batch['embeddings'].to(accelerator.device)
                    for i in range(mask_embeddings.shape[0]):
                        for j in range(top64_label.shape[1]):
                            label = top64_label[i,j].item()
                            if label in centroid_dict:
                                mask_embeddings[i,j,:] += torch.tensor(centroid_dict[label]).to(accelerator.device)
                            else:
                                print(f"Warning: label {label} not found in centroid_dict")
                    mask_embeddings = mask_embeddings.to(dtype=weight_dtype)

                    # Run IP-Adapter/UNet to predict noise residual
                    model_pred = ip_adapter(
                        latents,
                        t,
                        encoder_hidden_states=batch["prompt_ids"],
                        added_cond_kwargs=batch["unet_added_conditions"],
                        down_block_additional_residuals=[sample.to(dtype=weight_dtype) for sample in down_block_res_samples],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                        mask_embeddings=mask_embeddings,
                    )

                    # Update latent using scheduler step
                    latents = noise_scheduler.step(
                        model_output=model_pred,
                        timestep=t,
                        sample=latents,
                    ).prev_sample

                predict_dir = os.path.join(args.test_result_dir, "gen_result")
                shadowfree_dir = os.path.join(args.test_result_dir, "gt_shadowfree_img")
                mask_dir = os.path.join(args.test_result_dir, "gt_object_mask")
                
                os.makedirs(predict_dir, exist_ok=True)
                os.makedirs(shadowfree_dir, exist_ok=True)
                os.makedirs(mask_dir, exist_ok=True)

                # Decode final latent to image
                latents = latents / vae.config.scaling_factor
                decoded = vae.decode(latents).sample
                decoded = (decoded.clamp(-1, 1) + 1) / 2
                decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()
                decoded = (decoded * 255).astype(np.uint8)[0]
                filename = os.path.join(predict_dir, f"{batch_name[0]}_{repeat}.jpg")
                Image.fromarray(decoded).save(filename)


                shadowfree_img = batch["pixel_values"][0].cpu().permute(1, 2, 0).numpy()
                shadowfree_img = ((np.clip((shadowfree_img + 1) / 2, 0, 1)) * 255).astype(np.uint8)
                shadowfree_path = os.path.join(shadowfree_dir, f"{batch_name[0]}_{repeat}.png")
                Image.fromarray(shadowfree_img).save(shadowfree_path)

                object_mask_tensor = batch["conditioning_pixel_values"][0, -1, :, :].cpu()
                object_mask = (object_mask_tensor.numpy().clip(0, 1) * 255).astype(np.uint8)
                mask_path = os.path.join(mask_dir, f"{batch_name[0]}_{repeat}.png")
                Image.fromarray(object_mask).save(mask_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
