"""
Evolink Image Input Node for ComfyUI

Uploads local images and converts them to publicly accessible URLs.
Max 14 images, 20MB per image. Formats: jpeg, jpg, png, webp.
"""

import os
import hashlib
import time as time_module
from typing import Tuple

import torch
import numpy as np
from PIL import Image


class EvolinkImageInputNode:
    """
    Evolink Image Input - Upload local images to generate public URLs

    Accepts multiple image inputs and converts them to publicly
    accessible URLs via ComfyUI's public address.
    """

    PUBLIC_BASE_URL = "https://comfyui.lanqiu.tech"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {},
            "optional": {},
        }

        # Create up to 14 optional image slots
        for i in range(1, 15):
            inputs["optional"][f"image{i}"] = ("IMAGE", {"default": None})

        inputs["optional"]["prefix"] = ("STRING", {"default": "evolink", "multiline": False})

        return inputs

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_urls", "image_count")
    FUNCTION = "upload_images"
    CATEGORY = "image"
    OUTPUT_NODE = True

    # Class-level output directory
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    def _save_tensor_to_file(self, image_tensor: torch.Tensor, output_dir: str, prefix: str, index: int) -> str:
        """Save image tensor to file and return the file path"""
        # image_tensor shape: (B, H, W, C) or (H, W, C)
        if image_tensor.dim() == 4:
            # Has batch dimension, take first
            img = image_tensor[0]
        else:
            img = image_tensor

        # Convert tensor to PIL Image
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)

        # Generate unique filename
        img_bytes = img_np.tobytes()
        timestamp = int(time_module.time() * 1000) % 100000
        hash_suffix = hashlib.md5(img_bytes[:1024]).hexdigest()[:6]
        filename = f"{prefix}_{timestamp}_{index:02d}_{hash_suffix}.png"
        filepath = os.path.join(output_dir, filename)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Save image
        pil_img.save(filepath)

        return filepath

    def _get_public_url(self, filepath: str, public_base: str) -> str:
        """Convert local file path to public URL"""
        filename = os.path.basename(filepath)
        return f"{public_base}/{filename}"

    def upload_images(self, **kwargs) -> Tuple[str, str]:
        """
        Convert image tensors to publicly accessible URLs

        Args:
            image1...image14: Optional image tensors from upstream nodes
            prefix: Filename prefix for saved images

        Returns:
            image_urls: Newline-separated list of public URLs
            image_count: Number of images successfully converted
        """
        # Extract prefix and collect non-None images
        prefix = kwargs.pop("prefix", "evolink")
        image_count = 0
        saved_urls = []

        # Collect all image inputs
        image_keys = sorted([k for k in kwargs.keys() if k.startswith("image")])
        for key in image_keys:
            img_tensor = kwargs[key]
            if img_tensor is not None:
                try:
                    filepath = self._save_tensor_to_file(img_tensor, self.OUTPUT_DIR, prefix, image_count)
                    url = self._get_public_url(filepath, self.PUBLIC_BASE_URL)
                    saved_urls.append(url)
                    image_count += 1
                except Exception as e:
                    print(f"[EvolinkImageInput] Failed to save {key}: {e}")

        if not saved_urls:
            return ("", "0")

        # Join URLs with newline
        url_string = "\n".join(saved_urls)

        print(f"[EvolinkImageInput] Saved {len(saved_urls)} images to {self.OUTPUT_DIR}")
        print(f"[EvolinkImageInput] Public URLs: {url_string}")

        return (url_string, str(len(saved_urls)))


# Node registration
NODE_CLASS_MAPPINGS = {
    "EvolinkImageInput": EvolinkImageInputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EvolinkImageInput": "Evolink Image Input",
}
