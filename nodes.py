import numpy as np
import torch
from PIL import Image


def _tensor_to_np_gray(tensor: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE tensor [B,H,W,C] → 2D grayscale float32 [H,W] 0-1."""
    img = tensor[0].cpu().numpy()          # [H,W,C]
    if img.shape[2] == 1:
        gray = img[:, :, 0]
    else:
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return gray.astype(np.float32)


def _np_gray_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """2D grayscale float32 [H,W] → ComfyUI IMAGE tensor [1,H,W,3]."""
    arr = arr.clip(0.0, 1.0)
    rgb = np.stack([arr, arr, arr], axis=-1)   # [H,W,3]
    return torch.from_numpy(rgb).unsqueeze(0)  # [1,H,W,3]


def _resize_to(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    pil = pil.resize((w, h), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


def freq_stripe_conv(natural: np.ndarray, stripe: np.ndarray) -> np.ndarray:
    """
    주파수 도메인 처리:
      F_n = FFT(natural)          — 자연 이미지 스펙트럼
      F_s = FFT(stripe)           — 줄무늬 스펙트럼 (줄무늬 주파수 피크 포함)
      F_result = F_n * F_s        — 주파수 도메인 곱 (공간 도메인 원형 컨볼루션과 동치)
      result   = Re(IFFT(F_result)) → 정규화

    컨볼루션 정리: IFFT(F_n ⊛ F_s) = natural × stripe  (스케일 인수 포함)
    즉, 스펙트럼 콘볼루션의 역푸리에 변환 = 공간 도메인 픽셀 곱.
    여기서는 FFT 도메인에서 직접 연산 후 IFFT하여 기하학적 합성을 생성한다.
    """
    H, W = natural.shape

    if natural.shape != stripe.shape:
        stripe = _resize_to(stripe, H, W)

    # 1. 주파수 도메인으로 변환
    F_n = np.fft.fft2(natural)   # 자연 이미지 복소 스펙트럼
    F_s = np.fft.fft2(stripe)    # 줄무늬 복소 스펙트럼 (고주파 피크)

    # 2. 주파수 도메인에서 두 스펙트럼을 공간 신호처럼 취급하여 직접 2D 컨볼루션
    #    F_n ⊛ F_s : F_result[u,v] = Σ_{k,l} F_n[k,l] * F_s[u-k, v-l]
    from scipy.signal import fftconvolve
    F_result = fftconvolve(F_n, F_s, mode='same')

    # 3. 역 푸리에 변환 (공간 도메인으로 복원), 실수부 추출
    result = np.real(np.fft.ifft2(F_result))

    # 4. 출력 정규화 [0, 1] — 스펙트럼 연산 후 필수
    r_min, r_max = result.min(), result.max()
    if r_max > r_min:
        result = (result - r_min) / (r_max - r_min)
    else:
        result = np.zeros_like(result)

    return result.astype(np.float32)


class StripeBlendNode:
    """자연 이미지 스펙트럼과 줄무늬 피크의 주파수 도메인 컨볼루션 → 역푸리에 변환"""

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
        natural = _tensor_to_np_gray(natural_image)
        stripe  = _tensor_to_np_gray(stripe_image)
        result  = freq_stripe_conv(natural, stripe)
        return (_np_gray_to_tensor(result),)


NODE_CLASS_MAPPINGS = {
    "StripeBlend": StripeBlendNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StripeBlend": "Stripe Blend (Freq Conv)",
}
