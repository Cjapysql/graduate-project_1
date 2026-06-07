# main_caption.py
# 단일 폴더 내 이미지 전체 Gemini 캡셔닝 처리

import os
import sys
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from PIL import Image

from google import genai

# UTF-8 출력 설정
sys.stdout.reconfigure(encoding="utf-8")

# =========================
# env 로드
# =========================
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY가 설정되지 않았습니다")

client = genai.Client(api_key=API_KEY)

# =========================
# 시스템 프롬프트
# =========================
SYSTEM_INSTRUCTION = """
출력은 반드시 다음 네 줄로만 구성하십시오: description:, outfit:, items:, environment:

당신은 이미지를 시각적 메타데이터로 변환하는 분석 시스템입니다.
출력은 반드시 description, outfit, items, environment 네 개의 필드를 포함해야 하며, 각 필드는 필드명과 콜론 뒤에 이어지는 단일 문장으로 작성하십시오. 필드명 외의 추가 문장, 번호, 목록, 불필요한 기호는 사용하지 마십시오.

description은 장면 전체를 이해할 수 있도록 충분히 자세한 서술형 문장으로 작성하십시오. 인물의 외형, 표정, 자세, 시선, 주변 사물과의 관계, 공간의 분위기가 자연스럽게 연결되도록 하며, 감정 단어 대신 입술의 모양, 눈매, 몸의 긴장도처럼 관측 가능한 물리적 특징을 중심으로 묘사하십시오.

outfit은 인물이 착용한 의복과 액세서리를 색상과 형태 중심으로 간결하게 요약하십시오. 장면 설명이나 행동 묘사는 포함하지 마십시오.

items는 이미지에 등장하는 주요 사물과 생물을 빠짐없이 포함하되, 필요할 경우 인물과의 물리적 위치 관계를 반영하십시오.

environment는 배경이 되는 공간의 성격을 요약하여 서술하십시오. 장식 여부, 색상, 조명 상태, 공간의 단순함이나 밀도를 포함하십시오.

모든 필드는 이미지에서 직접 관측 가능한 정보만을 사용해야 하며, 매체를 지칭하는 단어, 텍스트 내용, 말풍선, 서명, 로고, 상징적 기호는 서술에서 제외하십시오.
"""

# =========================
# 자연 정렬 함수
# =========================
def natural_sort_key(path):
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", path.name)
    ]

# =========================
# 경로 설정
# =========================
SRC_DIR = Path(__file__).parent
IMAGE_DIR = SRC_DIR / "im"

CAPTION_OUTPUT_DIR = SRC_DIR / "json_data_caption"
CAPTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

print(f"\n🖼 총 {len(image_files)}개 이미지 캡셔닝 시작\n")

success_count = 0
failed_count = 0

# =========================
# 캡셔닝 처리
# =========================
for img_path in tqdm(image_files, desc="🧠 Gemini 캡셔닝"):
    image_file = img_path.name

    try:
        img = Image.open(img_path)

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[SYSTEM_INSTRUCTION, img],
            config={
                "response_mime_type": "text/plain",
                "max_output_tokens": 400
            }
        )

        text = response.text.strip()

        # =========================
        # 필드 파싱
        # =========================
        result = {
            "image_file": image_file,
            "description": "",
            "outfit": "",
            "items": "",
            "environment": ""
        }

        for line in text.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                if key in result:
                    result[key] = value.strip()

        out_path = CAPTION_OUTPUT_DIR / f"{img_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        success_count += 1

    except Exception as e:
        failed_count += 1
        print(f"\n❌ 실패: {image_file} | {e}")
        continue

# =========================
# 통계 출력
# =========================
print("\n✅ 캡셔닝 완료")
print(f"   ✓ JSON 생성: {success_count}개")
print(f"   ✗ 실패: {failed_count}개")
print(f"📂 결과 위치: {CAPTION_OUTPUT_DIR}")
