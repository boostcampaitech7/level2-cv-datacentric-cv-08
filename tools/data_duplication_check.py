import os
import json
from PIL import Image
import imagehash
from tqdm import tqdm

def get_image_files(folder_path):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]


def get_image_hash(image_path):
    img = Image.open(image_path)
    return imagehash.phash(img)


def find_duplicates_by_hash(original_folder, additional_folder):
    original_images = get_image_files(original_folder)
    additional_images = get_image_files(additional_folder)
    

    print("원본 이미지 해시 정보 추출 중...")
    original_hashes = {img_path: get_image_hash(img_path) for img_path in tqdm(original_images, desc="원본 이미지 처리")}
    

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
    
    for dup in duplicates:
        os.remove(dup)

    print(f"Duplicates removed: {len(duplicates)}")

if __name__ == "__main__":
    original_folder = "./original_images"
    additional_folder = "./new_images"

    find_duplicates_by_hash(original_folder, additional_folder)
