import numpy as np
import torch


def _tensor_to_np_gray(tensor: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE tensor [B,H,W,C] → 2D grayscale float32 [H,W] 0-1."""
    # 첫 번째 배치만 사용
    img = tensor[0].cpu().numpy()          # [H,W,C]
    if img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        # 가중 평균으로 grayscale 변환
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return gray.astype(np.float32)


def _np_gray_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """2D grayscale float32 [H,W] → ComfyUI IMAGE tensor [1,H,W,3]."""
    arr = arr.clip(0.0, 1.0)
    rgb = np.stack([arr, arr, arr], axis=-1)   # [H,W,3]
    return torch.from_numpy(rgb).unsqueeze(0)  # [1,H,W,3]


class StripeBlendNode:
    """자연 이미지 × 줄무늬 이미지 → 기하학적 합성 이미지"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "natural_image": ("IMAGE",),
                "stripe_image":  ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "blend"
    CATEGORY = "image/postprocessing"

    def blend(self, natural_image: torch.Tensor, stripe_image: torch.Tensor):
        natural = _tensor_to_np_gray(natural_image)   # [H,W]
        stripe  = _tensor_to_np_gray(stripe_image)    # [H,W]

        # 크기가 다르면 stripe를 natural 크기에 맞춤
        if natural.shape != stripe.shape:
            from PIL import Image
            h, w = natural.shape
            stripe_pil = Image.fromarray((stripe * 255).clip(0, 255).astype(np.uint8))
            stripe_pil = stripe_pil.resize((w, h), Image.LANCZOS)
            stripe = np.array(stripe_pil, dtype=np.float32) / 255.0

        result = (natural * stripe).astype(np.float32)
        return (_np_gray_to_tensor(result),)


NODE_CLASS_MAPPINGS = {
    "StripeBlend": StripeBlendNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StripeBlend": "Stripe Blend (Natural × Stripe)",
}
