"""
Evolink GPT Image 2 Node for ComfyUI

Integrates Evolink.ai GPT Image 2 API with ComfyUI.
Supports text-to-image and reference-image-assisted generation.

API Documentation: https://evolink.ai/zh/gpt-image-2
"""

import asyncio
import json
import io
import numpy as np
import urllib.request
import urllib.error
from typing import Tuple, List, Optional

import torch
from PIL import Image


class EvolinkGPTImage2Node:
    """
    Evolink GPT Image 2 - Image Generation Node

    Calls the Evolink.ai API to generate images using GPT Image 2 model.
    Supports text-to-image and image-to-image generation.

    API Endpoints:
    - POST /v1/images/generations (create task)
    - GET /v1/tasks/{task_id} (query status)
    """

    API_BASE = "https://api.evolink.ai"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["gpt-image-2", "gpt-image-2-beta"], {"default": "gpt-image-2"}),
            },
            "optional": {
                "size": (["auto", "1:1", "1:2", "2:1", "1:3", "3:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9", "9:21"], {"default": "auto"}),
                "resolution": (["1K", "2K", "4K"], {"default": "1K"}),
                "quality": (["low", "medium", "high"], {"default": "medium"}),
                "n": ("INT", {"default": 1, "min": 1, "max": 10}),
                "image_urls": ("STRING", {"default": "", "multiline": True}),
                "callback_url": ("STRING", {"default": "", "multiline": False}),
                "poll_interval": ("INT", {"default": 3, "min": 1, "max": 60}),
                "max_polls": ("INT", {"default": 100, "min": 1, "max": 500}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("image", "task_id", "status_info")
    FUNCTION = "generate_image"
    CATEGORY = "image/generation"
    OUTPUT_NODE = True

    def _make_request(self, method: str, path: str, data: Optional[dict] = None, api_key: str = "") -> dict:
        """Make HTTP request to Evolink API"""
        url = f"{self.API_BASE}{path}"

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        request_data = None
        if data:
            request_data = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(url, data=request_data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                response_body = response.read().decode("utf-8")
                return json.loads(response_body)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            try:
                error_json = json.loads(error_body)
                error_msg = error_json.get("error", {}).get("message", error_body)
            except:
                error_msg = error_body
            raise Exception(f"API Error {e.code}: {error_msg}")
        except urllib.error.URLError as e:
            raise Exception(f"Network Error: {e.reason}")

    async def _poll_task_status(self, task_id: str, api_key: str, poll_interval: int, max_polls: int) -> dict:
        """Poll task status until completion or failure"""
        for i in range(max_polls):
            result = self._make_request("GET", f"/v1/tasks/{task_id}", api_key=api_key)

            status = result.get("status", "unknown")

            if status == "completed":
                return result
            elif status == "failed":
                error_info = result.get("error", "Unknown error")
                raise Exception(f"Task failed: {error_info}")

            # Wait before next poll
            await asyncio.sleep(poll_interval)

        raise Exception(f"Task polling timeout after {max_polls} attempts")

    def _download_image(self, url: str) -> Image.Image:
        """Download image from URL and return PIL Image"""
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                image_data = response.read()
                return Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            raise Exception(f"Failed to download image from {url}: {str(e)}")

    def _pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to ComfyUI tensor format (B, H, W, C)"""
        image_np = np.array(image).astype(np.float32) / 255.0
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image_np)[None,]
        return tensor

    async def generate_image_async(self, api_key: str, prompt: str, model: str,
                                     size: str, resolution: str, quality: str,
                                     n: int, image_urls: str, callback_url: str,
                                     poll_interval: int, max_polls: int) -> Tuple[torch.Tensor, str, str]:
        """Async version of image generation"""

        # Parse image URLs
        image_url_list = []
        if image_urls.strip():
            image_url_list = [url.strip() for url in image_urls.strip().split("\n") if url.strip()]

        # Build request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "resolution": resolution,
            "quality": quality,
            "n": n,
        }

        if image_url_list:
            payload["image_urls"] = image_url_list

        if callback_url.strip():
            payload["callback_url"] = callback_url.strip()

        # Create task
        result = self._make_request("POST", "/v1/images/generations", data=payload, api_key=api_key)

        task_id = result.get("id", "")
        status = result.get("status", "pending")
        task_info = result.get("task_info", {})

        status_info = f"Task: {task_id} | Status: {status} | ETA: {task_info.get('estimated_time', 'N/A')}s"

        if status == "completed":
            results = result.get("results", [])
        elif status == "failed":
            error_info = result.get("error", "Unknown error")
            raise Exception(f"Task failed immediately: {error_info}")
        else:
            # Poll for completion
            final_result = await self._poll_task_status(task_id, api_key, poll_interval, max_polls)
            results = final_result.get("results", [])
            status_info = f"Completed: {task_id} | Images: {len(results)}"

        # Download images
        pil_images = []
        for idx, image_url in enumerate(results):
            try:
                img = self._download_image(image_url)
                pil_images.append(img)
            except Exception as e:
                print(f"Warning: Failed to download image {idx}: {e}")

        # Convert to tensors and stack
        if not pil_images:
            raise Exception("No images were successfully downloaded")

        # Convert each PIL image to tensor
        tensors = []
        for pil_img in pil_images:
            img_np = np.array(pil_img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(img_np)
            tensors.append(tensor)

        # Stack into batch (B, H, W, C)
        image_batch = torch.stack(tensors, dim=0)

        return image_batch, task_id, status_info

    def generate_image(self, api_key: str, prompt: str, model: str,
                       size: str = "auto", resolution: str = "1K", quality: str = "medium",
                       n: int = 1, image_urls: str = "", callback_url: str = "",
                       poll_interval: int = 3, max_polls: int = 100,
                       unique_id: str = "") -> Tuple[torch.Tensor, str, str]:
        """Synchronous wrapper for async generation"""

        try:
            # Check if we're already in an async context
            loop = asyncio.get_running_loop()
            # Already in async context - use thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self._generate_image_sync,
                    api_key, prompt, model, size, resolution, quality,
                    n, image_urls, callback_url, poll_interval, max_polls
                )
                return future.result()
        except RuntimeError:
            # No running loop, safe to create one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            return loop.run_until_complete(
                self.generate_image_async(
                    api_key, prompt, model, size, resolution, quality,
                    n, image_urls, callback_url, poll_interval, max_polls
                )
            )

    def _generate_image_sync(self, api_key: str, prompt: str, model: str,
                             size: str, resolution: str, quality: str,
                             n: int, image_urls: str, callback_url: str,
                             poll_interval: int, max_polls: int) -> Tuple[torch.Tensor, str, str]:
        """Synchronous version for thread pool execution"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.generate_image_async(
                    api_key, prompt, model, size, resolution, quality,
                    n, image_urls, callback_url, poll_interval, max_polls
                )
            )
        finally:
            loop.close()


# Node registration
NODE_CLASS_MAPPINGS = {
    "EvolinkGPTImage2": EvolinkGPTImage2Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EvolinkGPTImage2": "Evolink GPT Image 2",
}
