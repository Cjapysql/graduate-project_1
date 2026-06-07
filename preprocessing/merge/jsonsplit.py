import json
import os

def split_json_by_filename(input_file, output_dir='split_jsons'):
    """
    하나의 큰 JSON 파일을 읽어 각 항목을 개별 JSON 파일로 분할합니다.
    """
    # 1. 출력 디렉토리 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 2. 원본 JSON 파일 로드
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"에러: {input_file} 파일을 찾을 수 없습니다.")
        return

    # 3. 데이터 순회 및 개별 저장
    count = 0
    for img_filename, caption in data.items():
        # 확장자 변경 (예: .png -> .json)
        base_name = os.path.splitext(img_filename)[0]
        json_filename = f"{base_name}.json"
        
        # 저장할 데이터 구조 생성
        individual_data = {
            "image_file": img_filename,
            "caption": caption
        }
        
        # 파일 저장
        file_path = os.path.join(output_dir, json_filename)
        with open(file_path, 'w', encoding='utf-8') as out_f:
            json.dump(individual_data, out_f, ensure_ascii=False, indent=4)
        
        count += 1

    print(f"성공: 총 {count}개의 파일이 '{output_dir}' 폴더에 생성되었습니다.")

# 실행 코드
if __name__ == "__main__":
    # 파일명이 다를 경우 아래 이름을 수정하세요.
    split_json_by_filename('webtoon_captions_results.json')