#!/usr/bin/env python3
"""
자연 이미지 × 줄무늬 이미지 → 기하학적 합성 이미지
사용법: python stripe_blend.py <자연이미지> <줄무늬이미지> [출력파일]
줄무늬 이미지를 생략하면 자동 생성합니다.
"""

import sys
import numpy as np
from PIL import Image
import argparse


def load_gray(path: str, size: tuple | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


def generate_stripe(width: int, height: int, frequency: float = 0.05, angle_deg: float = 45.0) -> np.ndarray:
    """사인파 줄무늬 패턴을 생성합니다."""
    angle = np.radians(angle_deg)
    x = np.arange(width)
    y = np.arange(height)
    xx, yy = np.meshgrid(x, y)
    proj = xx * np.cos(angle) + yy * np.sin(angle)
    stripe = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * proj)
    return stripe.astype(np.float32)


def fft_magnitude(img: np.ndarray) -> np.ndarray:
    """로그 스케일 FFT 크기 스펙트럼을 반환합니다."""
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log1p(np.abs(F))
    mag = (mag / mag.max() * 255).astype(np.uint8)
    return mag


def blend(natural: np.ndarray, stripe: np.ndarray) -> np.ndarray:
    """픽셀 단위 곱셈 합성."""
    if natural.shape != stripe.shape:
        h, w = natural.shape
        stripe_img = Image.fromarray((stripe * 255).astype(np.uint8))
        stripe_img = stripe_img.resize((w, h), Image.LANCZOS)
        stripe = np.array(stripe_img, dtype=np.float32) / 255.0
    return (natural * stripe).astype(np.float32)


def make_panel(images: list[np.ndarray], labels: list[str]) -> Image.Image:
    """여러 이미지를 가로로 나열한 패널을 만듭니다."""
    from PIL import ImageDraw, ImageFont

    h, w = images[0].shape
    pad = 4
    label_h = 24
    panel_w = (w + pad) * len(images) - pad
    panel_h = h + label_h
    panel = Image.new("L", (panel_w, panel_h), 200)

    for i, (arr, label) in enumerate(zip(images, labels)):
        img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        x = i * (w + pad)
        panel.paste(img, (x, label_h))
        draw = ImageDraw.Draw(panel)
        draw.text((x + 4, 4), label, fill=0)

    return panel


def main():
    parser = argparse.ArgumentParser(description="자연 이미지 × 줄무늬 → 기하학적 합성")
    parser.add_argument("natural", help="자연 이미지 경로")
    parser.add_argument("stripe", nargs="?", default=None, help="줄무늬 이미지 경로 (없으면 자동 생성)")
    parser.add_argument("output", nargs="?", default="output_blend.png", help="출력 파일 경로")
    parser.add_argument("--freq", type=float, default=0.05, help="자동 생성 줄무늬 주파수 (기본 0.05)")
    parser.add_argument("--angle", type=float, default=45.0, help="자동 생성 줄무늬 각도 (기본 45도)")
    parser.add_argument("--fft", action="store_true", help="FFT 스펙트럼 패널도 저장")
    args = parser.parse_args()

    # 자연 이미지 로드
    natural = load_gray(args.natural)
    h, w = natural.shape

    # 줄무늬 이미지 로드 또는 생성
    if args.stripe:
        stripe = load_gray(args.stripe, size=(w, h))
    else:
        print(f"줄무늬 자동 생성: freq={args.freq}, angle={args.angle}°")
        stripe = generate_stripe(w, h, frequency=args.freq, angle_deg=args.angle)

    # 합성
    result = blend(natural, stripe)

    # 결과 저장
    out_img = Image.fromarray((result * 255).clip(0, 255).astype(np.uint8))
    out_img.save(args.output)
    print(f"저장됨: {args.output}")

    # FFT 패널 저장 (선택)
    if args.fft:
        fft_natural = fft_magnitude(natural).astype(np.float32) / 255.0
        fft_stripe = fft_magnitude(stripe).astype(np.float32) / 255.0
        fft_result = fft_magnitude(result).astype(np.float32) / 255.0

        # 상단: 원본 × 줄무늬 = 합성
        top_panel = make_panel(
            [natural, stripe, result],
            ["Natural", "Stripe", "Natural × Stripe"]
        )
        # 하단: FFT 시각화
        bot_panel = make_panel(
            [fft_natural, fft_stripe, fft_result],
            ["FFT(Natural)", "FFT(Stripe)", "FFT(Result)"]
        )

        full_h = top_panel.height + bot_panel.height + 8
        full = Image.new("L", (top_panel.width, full_h), 200)
        full.paste(top_panel, (0, 0))
        full.paste(bot_panel, (0, top_panel.height + 8))

        fft_path = args.output.replace(".png", "_fft_panel.png")
        full.save(fft_path)
        print(f"FFT 패널 저장됨: {fft_path}")


if __name__ == "__main__":
    main()
