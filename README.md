# Comfyui_StrypeConv

A ComfyUI custom node that blends a natural image with a stripe pattern image using pixel-wise multiplication, producing a geometric composite output.

## How It Works

The node multiplies each pixel of the natural image by the corresponding pixel of the stripe image:

```
output = natural_image × stripe_image
```

This is equivalent to convolution in the frequency domain — the FFT of the result is the convolution of the two input spectra.

![Example: natural × stripe = blended output](example.png)

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/yourname/Comfyui_StrypeConv
```

Then restart ComfyUI.

No additional dependencies are required beyond what ComfyUI already provides (`numpy`, `torch`, `Pillow`).

## Node

### Stripe Blend (Natural × Stripe)

Found under **image → postprocessing** in the node browser.

| Port | Type | Description |
|------|------|-------------|
| `natural_image` | IMAGE (input) | The source image (photo, render, etc.) |
| `stripe_image` | IMAGE (input) | The stripe pattern to multiply with |
| `blended_image` | IMAGE (output) | Pixel-wise product of the two inputs |

**Notes:**
- Both inputs are converted to grayscale before multiplication.
- If the two images differ in size, the stripe image is resized to match the natural image using Lanczos resampling.
- The output is a grayscale image returned as an RGB tensor compatible with all standard ComfyUI image nodes.

## Standalone CLI

A command-line version is also included for use outside of ComfyUI.

```bash
# Use an existing stripe image
python stripe_blend.py natural.jpg stripe.jpg output.png

# Auto-generate a sine-wave stripe (frequency=0.05, angle=45°)
python stripe_blend.py natural.jpg output.png --freq 0.05 --angle 45

# Also save an FFT spectrum panel
python stripe_blend.py natural.jpg stripe.jpg output.png --fft
```

| Argument | Description | Default |
|----------|-------------|---------|
| `natural` | Path to the natural image | required |
| `stripe` | Path to the stripe image | optional |
| `output` | Output file path | `output_blend.png` |
| `--freq` | Stripe frequency when auto-generating | `0.05` |
| `--angle` | Stripe angle in degrees when auto-generating | `45` |
| `--fft` | Save an additional FFT spectrum panel | off |

## License

MIT
