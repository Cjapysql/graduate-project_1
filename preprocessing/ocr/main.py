# python src/main.py

import cv2
import os
import json
import sys
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

from detector import run_detector
from ocr_clients.clova import ClovaOCR

# UTF-8 출력 설정
sys.stdout.reconfigure(encoding='utf-8')

# env 로드
load_dotenv()

INVOKE_URL = os.getenv("CLOVA_OCR_INVOKE_URL")
SECRET_KEY = os.getenv("CLOVA_OCR_SECRET")

if not INVOKE_URL or not SECRET_KEY:
    raise RuntimeError("CLOVA OCR 환경변수가 설정되지 않았습니다")

# 자연스러운 정렬 함수
def natural_sort_key(path):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', str(path.name))]

# =========================
# 경로 설정
# =========================
SRC_DIR = Path(__file__).parent
IMAGE_DIR = SRC_DIR / "images" / "total_processed"

# ⭐ OCR 결과 저장 폴더
OCR_OUTPUT_DIR = SRC_DIR / "json_data_ocr"
OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not IMAGE_DIR.exists():
    raise RuntimeError(f"❌ 폴더가 존재하지 않습니다: {IMAGE_DIR}")

print(f"✓ 이미지 폴더 확인: {IMAGE_DIR}")
print(f"✓ OCR 결과 저장 폴더: {OCR_OUTPUT_DIR}")

# OCR 객체 생성
ocr = ClovaOCR(
    invoke_url=INVOKE_URL,
    secret_key=SECRET_KEY
)

# 이미지 리스트
all_image_files = sorted(IMAGE_DIR.glob("*.png"), key=natural_sort_key)

if not all_image_files:
    raise RuntimeError(f"{IMAGE_DIR}에 PNG 이미지가 없습니다")

TEST_LIMIT = None  # 테스트 시 숫자로 변경

image_files = all_image_files[:TEST_LIMIT] if TEST_LIMIT else all_image_files
print(f"📁 총 {len(image_files)}개 이미지 처리 예정\n")

# =========================
# OCR 처리
# =========================
success_count = 0
failed_count = 0
no_blocks_count = 0

for img_path in tqdm(image_files, desc="🔍 OCR 처리 중"):
    image_file = img_path.name

    # 이미지 로드
    try:
        from PIL import Image
        import numpy as np

        pil_img = Image.open(img_path)
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        failed_count += 1
        continue

    # 텍스트 탐지
    try:
        blocks = run_detector(str(img_path))
    except Exception:
        failed_count += 1
        continue

    if not blocks:
        no_blocks_count += 1
        continue

    ocr_blocks = []

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

        # 🔹 confidence 제거 + text flatten
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

    # =========================
    # ⭐ 이미지 1장 = JSON 1개 저장
    # =========================
    output_json = {
        "image_file": image_file,
        "ocr": ocr_blocks
    }

    out_path = OCR_OUTPUT_DIR / f"{Path(image_file).stem}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    success_count += 1

# =========================
# 통계 출력
# =========================
print("\n✅ OCR 처리 완료")
print(f"   ✓ 성공: {success_count}개")
print(f"   ✗ 실패: {failed_count}개")
print(f"   ○ 텍스트 없음: {no_blocks_count}개")
print(f"📂 결과 위치: {OCR_OUTPUT_DIR}")

