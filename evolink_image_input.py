"""
Evolink Image Input Node for ComfyUI

Uploads local images and converts them to publicly accessible URLs.
Max 14 images, 20MB per image. Formats: jpeg, jpg, png, webp.

Primary: Use ComfyUI output folder URL
Fallback: Upload to imgbb if URL is not publicly accessible
"""

import os
import hashlib
import time as time_module
import urllib.request
import urllib.error
import urllib.parse
import base64
import json
from typing import Tuple, Optional

import torch
import numpy as np
from PIL import Image


class EvolinkImageInputNode:
    """
    Evolink Image Input - Upload local images to generate public URLs

    Accepts multiple image inputs and converts them to publicly
    accessible URLs via ComfyUI's public address or imgbb fallback.
    """

    PUBLIC_BASE_URL = "https://comfyui.lanqiu.tech"

    # imgbb free API key (anonymous tier)
    IMGBB_API_KEY = "dcay4528303703a99a5c66894c356ab1"

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
        inputs["optional"]["force_imgbb"] = ("BOOLEAN", {"default": False})

        return inputs

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_urls", "image_count")
    FUNCTION = "upload_images"
    CATEGORY = "image"
    OUTPUT_NODE = True

    # Class-level output directory - ComfyUI's root output folder
    _node_dir = os.path.dirname(os.path.abspath(__file__))
    _comfyui_root = os.path.abspath(os.path.join(_node_dir, "..", ".."))
    OUTPUT_DIR = os.path.join(_comfyui_root, "output")

    def _save_tensor_to_file(self, image_tensor: torch.Tensor, output_dir: str, prefix: str, index: int) -> str:
        """Save image tensor to file and return the file path"""
        # image_tensor shape: (B, H, W, C) or (H, W, C)
        if image_tensor.dim() == 4:
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

    def _check_url_accessible(self, url: str, timeout: int = 5) -> bool:
        """Check if a URL is publicly accessible"""
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status == 200
        except Exception:
            return False

    def _upload_to_imgbb(self, filepath: str) -> Optional[str]:
        """Upload image to imgbb and return public URL"""
        try:
            with open(filepath, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            post_data = urllib.parse.urlencode({
                "key": self.IMGBB_API_KEY,
                "image": image_data,
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.imgbb.com/1/upload",
                data=post_data,
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                if result.get("success"):
                    return result["data"]["url"]
        except Exception as e:
            print(f"[EvolinkImageInput] imgbb upload failed: {e}")
        return None

    def upload_images(self, **kwargs) -> Tuple[str, str]:
        """
        Convert image tensors to publicly accessible URLs
        """
        prefix = kwargs.pop("prefix", "evolink")
        force_imgbb = kwargs.pop("force_imgbb", False)

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

                    # Check if URL is accessible, fallback to imgbb if not
                    if force_imgbb or not self._check_url_accessible(url):
                        print(f"[EvolinkImageInput] Output URL not accessible, uploading to imgbb...")
                        imgbb_url = self._upload_to_imgbb(filepath)
                        if imgbb_url:
                            url = imgbb_url
                            print(f"[EvolinkImageInput] imgbb URL: {url}")
                        else:
                            print(f"[EvolinkImageInput] WARNING: Using potentially inaccessible URL: {url}")
                    else:
                        print(f"[EvolinkImageInput] URL accessible: {url}")

                    if url.startswith("http"):
                        saved_urls.append(url)
                        image_count += 1
                except Exception as e:
                    print(f"[EvolinkImageInput] Failed to save {key}: {e}")

        if not saved_urls:
            return ("", "0")

        url_string = "\n".join(saved_urls)

        print(f"[EvolinkImageInput] Saved {len(saved_urls)} images")
        print(f"[EvolinkImageInput] URLs: {url_string}")

        return (url_string, str(len(saved_urls)))


# Node registration
NODE_CLASS_MAPPINGS = {
    "EvolinkImageInput": EvolinkImageInputNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EvolinkImageInput": "Evolink Image Input",
}
