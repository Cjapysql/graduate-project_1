import os
import json
import glob
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 경로 설정 ---
input_folder = "/data2/myeonghun/webtoon/caption_raw"
output_folder = "/data2/myeonghun/webtoon/caption_translated"
model_id = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

# 1. 모델 및 토크나이저 설정 (풀스펙 BF16)
print(f">>> {model_id} 모델을 BF16 모드로 로드합니다.")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # 양자화 없이 BF16 풀스펙 사용
    device_map="auto",          # CUDA_VISIBLE_DEVICES를 따름
    trust_remote_code=True
)

# 2. 번역 함수 정의
def translate_caption(eng_text):
    messages = [
        {"role": "system", "content": "너는 웹툰 캡션 전문 번역가야. 입력되는 영어 묘사들을 자연스럽게 한국어로 번역해줘. 설명 없이 번역 결과만 출력해."},
        {"role": "user", "content": f"번역해: {eng_text}"}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return result.strip()

# 3. 실행 로직
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

json_files = glob.glob(os.path.join(input_folder, "*.json"))
print(f">>> 총 {len(json_files)}개의 파일을 처리합니다.")

for file_path in tqdm(json_files, desc="BF16 번역 진행 중"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "caption" in data and data["caption"].strip():
            data["caption"] = translate_caption(data["caption"])
            
            file_name = os.path.basename(file_path)
            save_path = os.path.join(output_folder, file_name)

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
    
    except Exception as e:
        print(f"\n[Error] {file_path}: {e}")

print(f"\n>>> 작업 완료! 결과: {output_folder}")