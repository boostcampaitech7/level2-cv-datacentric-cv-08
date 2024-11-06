import os
import json
from PIL import Image
import imagehash
from tqdm import tqdm

# 이미지 파일 목록 가져오기
def get_image_files(folder_path):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]

# 이미지 해시 추출 함수
def get_image_hash(image_path):
    img = Image.open(image_path)
    return imagehash.phash(img)  # Perceptual Hash 계산

# 중복 검사 및 JSON 저장 함수
def find_duplicates_by_hash(original_folder, additional_folder, json_path):
    original_images = get_image_files(original_folder)
    additional_images = get_image_files(additional_folder)
    
    # 원본 이미지 해시 값 추출
    print("원본 이미지 해시 정보 추출 중...")
    original_hashes = {img_path: get_image_hash(img_path) for img_path in tqdm(original_images, desc="원본 이미지 처리")}
    
    # 추가 이미지 해시 값 추출 및 중복 확인
    print("추가 이미지 해시 정보 추출 중...")
    duplicates = []
    items = []
    additional_hashes = {}
    
    for img_path in tqdm(additional_images, desc="추가 이미지 처리"):
        img_hash = get_image_hash(img_path)
        is_duplicate = any(img_hash == orig_hash for orig_hash in original_hashes.values())
        
        if is_duplicate:
            duplicates.append(img_path)
        else:
            additional_hashes[img_path] = img_hash
            items.append({
                "id": os.path.basename(img_path),
                "hash": str(img_hash)
            })
    
    # 중복 이미지 삭제
    for dup in duplicates:
        os.remove(dup)
    
    # 결과 JSON 저장
    final_data = {
        "info": {},
        "items": items
    }
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(final_data, json_file, ensure_ascii=False, indent=4)

    print(f"Duplicates removed: {len(duplicates)}")
    print(f"JSON saved to {json_path}")

# 메인 실행
if __name__ == "__main__":
    original_folder = "./original_images"
    additional_folder = "./new_images"
    output_folder = "./pseudo_labels"
    json_path = os.path.join(output_folder, "annotations.json")

    # Pseudo 라벨 저장 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    find_duplicates_by_hash(original_folder, additional_folder, json_path)
