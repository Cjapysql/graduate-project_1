import os
# GPU 1번 사용 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import glob
import json
from tqdm import tqdm

# [1] 경로 및 모델 설정
INPUT_DIR = "/data2/myeonghun/webtoon/webtoon_image/webtoon_final_cuts/"
OUTPUT_DIR = "/data2/myeonghun/webtoon/data/"
MODEL_NAME = "fancyfeast/llama-joycaption-alpha-two-hf-llava"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# [2] 사용자 제공 프롬프트 (그대로 적용)
SYSTEM_PROMPT = ""
PROMPT = """Describe this scene as a real-world physical environment with extreme precision. Do not use any introductory phrases or words like "illustration", "digital drawing", "webtoon", or "artwork". Start directly with the physical description.

Provide an exhaustive breakdown of the characters, including their hair texture, eye shape, and every detail of their clothing such as seams and buttons. Describe facial expressions purely through physical movements like the curve of the lips or the angle of the eyebrows to capture the exact emotion for search retrieval.

Capture the entire background in detail, even the smallest props and distant objects; identify their material, color, and placement. Ignore all speech bubbles, on-screen text, and sound effects completely. Report only verifiable visual facts.

Output the entire description as a single, dense paragraph of plain text. Do not use any bullet points, numbers, or symbols like dashes or asterisks. Focus on high-density keywords that a user might search for."""

# [3] 모델 및 프로세서 로드
print(f"🚀 JoyCaption Alpha (Full) 모델 로딩 중 (GPU 1번): {MODEL_NAME}")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
llava_model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    torch_dtype="bfloat16", 
    device_map="auto"
)
llava_model.eval()

# [4] 이미지 파일 리스트 확보 (하위 폴더 재귀적 검색 적용)
# "**/*.png" 패턴과 recursive=True를 통해 모든 깊이의 폴더를 뒤집니다.
image_extensions = ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG")
image_files = []
for ext in image_extensions:
    # 하위 폴더 어디에 있든 해당 확장자를 가진 파일을 모두 찾습니다.
    search_pattern = os.path.join(INPUT_DIR, "**", f"*.{ext}")
    image_files.extend(glob.glob(search_pattern, recursive=True))

print(f"📂 하위 폴더 포함 총 {len(image_files)}개의 이미지를 발견했습니다.")

# [5] 처리 및 저장
for img_path in tqdm(image_files, desc="JoyCaption Processing"):
    file_name = os.path.basename(img_path)
    file_base_name = os.path.splitext(file_name)[0]
    output_json_path = os.path.join(OUTPUT_DIR, f"{file_base_name}.json")
    
    # 이미 처리된 파일은 스킵
    if os.path.exists(output_json_path):
        continue

    # 파일명 파싱 (1_002.png -> ep: 1, scene: 002)
    try:
        parts = file_base_name.split('_')
        episode_num = parts[0]
        scene_num = parts[1]
    except IndexError:
        episode_num = "unknown"
        scene_num = "unknown"

    try:
        with torch.no_grad():
            image = Image.open(img_path).convert("RGB")

            convo = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": PROMPT},
            ]
            
            convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[convo_string], images=[image], return_tensors="pt").to('cuda')
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

            generate_ids = llava_model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                use_cache=True,
            )[0]

            generate_ids = generate_ids[inputs['input_ids'].shape[1]:]
            caption = processor.tokenizer.decode(generate_ids, skip_special_tokens=True).strip()

            # [6] 결과 JSON 저장
            result_data = {
                "image_file": file_name,
                "episode": episode_num,
                "scene": scene_num,
                "caption": caption
            }
            
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=4)

    except Exception as e:
        print(f"⚠️ {file_name} 처리 중 오류 발생: {e}")

print(f"✅ 모든 작업 완료! 결과 폴더: {OUTPUT_DIR}")