# python main4.py
# 단일 폴더 내 이미지 전체 OCR 처리
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import cv2
import json
import sys
import re
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from detector import run_detector
from ocr_clients.clova import ClovaOCR

# UTF-8 출력 설정
sys.stdout.reconfigure(encoding='utf-8')

# =========================
# env 로드
# =========================
load_dotenv()

INVOKE_URL = os.getenv("CLOVA_OCR_INVOKE_URL")
SECRET_KEY = os.getenv("CLOVA_OCR_SECRET")

if not INVOKE_URL or not SECRET_KEY:
    raise RuntimeError("❌ CLOVA OCR 환경변수가 설정되지 않았습니다")

# =========================
# 자연 정렬 함수
# =========================
def natural_sort_key(path):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r'(\d+)', path.name)
    ]

# =========================
# 경로 설정
# =========================
SRC_DIR = Path(__file__).resolve().parent
WEBTOON_DIR = SRC_DIR.parent.parent
IMAGE_DIR = WEBTOON_DIR / "webtoon_image" / "im"

OCR_OUTPUT_DIR = SRC_DIR / "extra_data_ocr"
OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not IMAGE_DIR.exists():
    raise RuntimeError(f"❌ 이미지 폴더가 존재하지 않습니다: {IMAGE_DIR}")

# =========================
# 이미지 파일 수집
# =========================
image_files = sorted(
    IMAGE_DIR.glob("*.png"),
    key=natural_sort_key
)

if not image_files:
    raise RuntimeError("❌ 처리할 이미지가 없습니다")

print(f"\n🖼 총 {len(image_files)}개 이미지 OCR 처리 시작\n")

# OCR 객체 생성
ocr = ClovaOCR(
    invoke_url=INVOKE_URL,
    secret_key=SECRET_KEY
)

# =========================
# OCR 처리
# =========================
success_count = 0
failed_count = 0
no_blocks_count = 0

for img_path in tqdm(image_files, desc="🔍 OCR 처리"):
    image_file = img_path.name

    # 이미지 로드
    try:
        from PIL import Image
        import numpy as np

        pil_img = Image.open(img_path)
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        failed_count += 1
        continue

    # 텍스트 탐지
    try:
        blocks = run_detector(str(img_path))
    except Exception:
        failed_count += 1
        continue

    ocr_blocks = []

    if blocks:
        for block_idx, block in enumerate(blocks):
            x1, y1, x2, y2 = block.xyxy

            pad = 8
            h, w, _ = image.shape
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)

            crop = image[y1:y2, x1:x2]

            try:
                texts = ocr.run(crop)
            except Exception:
                continue

            merged_texts = [
                t["text"].strip()
                for t in texts
                if t.get("text")
            ]

            if not merged_texts:
                continue

            ocr_blocks.append({
                "block_number": block_idx,
                "text": " ".join(merged_texts)
            })
    else:
        no_blocks_count += 1

    # =========================
    # 이미지 1장 = JSON 1개 저장
    # =========================
    output_json = {
        "image_file": image_file,
        "ocr": ocr_blocks
    }

    out_path = OCR_OUTPUT_DIR / f"{img_path.stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    success_count += 1

# =========================
# 통계 출력
# =========================
print("\n✅ OCR 처리 완료")
print(f"   ✓ JSON 생성: {success_count}개")
print(f"   ✗ 실패: {failed_count}개")
print(f"   ○ 텍스트 없음: {no_blocks_count}개")
print(f"📂 결과 위치: {OCR_OUTPUT_DIR}")

