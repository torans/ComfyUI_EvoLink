# ComfyUI_Evolink

ComfyUI custom node for [Evolink.ai](https://evolink.ai) GPT Image 2 API integration.

## Features

- Text-to-image generation using GPT Image 2
- Image-to-image generation with reference images (up to 16 images)
- Configurable aspect ratio, resolution, and quality
- Async task polling with progress tracking
- Batch generation (up to 10 images per request)

## Installation

1. Clone or copy this repository to your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI_Evolink.git
```

Or simply copy the `ComfyUI_Evolink` folder to `custom_nodes/`.

2. Install dependencies (if not already installed):

```bash
pip install torch pillow
```

## API Key Setup

1. Sign up at [Evolink.ai](https://evolink.ai)
2. Get your API key from your dashboard
3. Enter the API key in the ComfyUI node interface

## Usage

1. **In ComfyUI**: Search for "Evolink GPT Image 2" in the node search
2. **Connect the node** to your workflow (e.g., connect to SaveImage node)
3. **Configure parameters**:
   - `api_key`: Your Evolink.ai API key
   - `prompt`: Image description (supports CJK, max 32,000 characters)
   - `model`: `gpt-image-2` (stable) or `gpt-image-2-beta`
   - `size`: Aspect ratio or "auto" for model decision
   - `resolution`: `1K`, `2K`, or `4K` (for ratio-based sizes)
   - `quality`: `low`, `medium`, or `high`
   - `n`: Number of images to generate (1-10)
   - `image_urls`: Reference images for img2img (one per line, up to 16)

## Output

- **IMAGE**: Generated image tensor (B, H, W, C)
- **task_id**: Evolink task ID for reference
- **status_info**: Status message with task details

## API Documentation

- Product Page: https://evolink.ai/zh/gpt-image-2
- API Docs: https://docs.evolink.ai

## License

MIT License
