"""
Evolink Image Input Node for ComfyUI

Uploads local images and converts them to publicly accessible URLs.
Max 14 images, 20MB per image. Formats: jpeg, jpg, png, webp.
"""

import os
import hashlib
import shutil
from typing import Tuple, List

import torch
import numpy as np
from PIL import Image
import comfy.utils


class EvolinkImageInputNode:
    """
    Evolink Image Input - Upload local images to generate public URLs

    Takes local image paths or tensors and converts them to publicly
    accessible URLs via ComfyUI's public address.
    """

    PUBLIC_BASE_URL = "https://comfyui.lanqiu.tech"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # Accept image tensor from upstream node
            },
            "optional": {
                "prefix": ("STRING", {"default": "evolink", "multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_urls", "image_count")
    FUNCTION = "upload_images"
    CATEGORY = "image"
    OUTPUT_NODE = True

    def _save_tensor_to_file(self, image_tensor: torch.Tensor, output_dir: str, prefix: str) -> List[str]:
        """Save image tensor to file and return the file path"""
        # image_tensor shape: (B, H, W, C) or (H, W, C)
        if image_tensor.dim() == 4:
            # Has batch dimension
            images = image_tensor
        else:
            # No batch dimension, add one
            images = image_tensor.unsqueeze(0)

        saved_paths = []
        for i in range(images.shape[0]):
            # Convert tensor to PIL Image
            img = images[i]
            # image_tensor is (H, W, C) in range [0, 1]
            img_np = (img.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Generate unique filename with timestamp to avoid collisions
            import time
            img_bytes = img_np.tobytes()
            timestamp = int(time.time() * 1000) % 100000
            hash_suffix = hashlib.md5(img_bytes[:1024]).hexdigest()[:6]
            filename = f"{prefix}_{timestamp}_{i:02d}_{hash_suffix}.png"
            filepath = os.path.join(output_dir, filename)

            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Save image
            pil_img.save(filepath)
            saved_paths.append(filepath)

        return saved_paths

    def _get_public_url(self, filepath: str, public_base: str) -> str:
        """Convert local file path to public URL"""
        # Get relative path from ComfyUI's input or output directory
        # ComfyUI typically serves from 'output' directory
        filename = os.path.basename(filepath)
        return f"{public_base}/{filename}"

    def upload_images(self, images: torch.Tensor, prefix: str = "evolink") -> Tuple[str, str]:
        """
        Convert image tensors to publicly accessible URLs

        Args:
            images: Image tensor from upstream node (B, H, W, C)
            prefix: Filename prefix for saved images

        Returns:
            image_urls: Newline-separated list of public URLs
            image_count: Number of images successfully converted
        """
        # Determine output directory
        # ComfyUI's output directory - use a subdir for evolink
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Save images to files
        saved_paths = self._save_tensor_to_file(images, output_dir, prefix)

        # Convert to public URLs
        public_urls = []
        for path in saved_paths:
            url = self._get_public_url(path, self.PUBLIC_BASE_URL)
            public_urls.append(url)

        # Join URLs with newline
        url_string = "\n".join(public_urls)
        count_string = str(len(public_urls))

        print(f"[EvolinkImageInput] Saved {len(public_urls)} images to {output_dir}")
        print(f"[EvolinkImageInput] Public URLs: {url_string}")

        return (url_string, count_string)


# Node registration
NODE_CLASS_MAPPINGS = {
    "EvolinkImageInput": EvolinkImageInputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EvolinkImageInput": "Evolink Image Input",
}
