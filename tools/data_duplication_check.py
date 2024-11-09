import os
import json
from PIL import Image
import imagehash
from tqdm import tqdm

def get_image_files(folder_path):
    """
    이미지 파일 목록을 가져옵니다
    
    Args:
        folder_path (str): 이미지 파일이 저장된 폴더 경로

    Returns:
        list: 폴더 내 이미지 파일의 전체 경로 리스트
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]


def get_image_hash(image_path):
    """
    이미지의 해시 값을 계산

    Args:
        image_path (str): 해시 값을 계산할 이미지 파일의 경로

    Returns:
        imagehash.ImageHash: 이미지의 Perceptual Hash 값
    """
    img = Image.open(image_path)
    return imagehash.phash(img)


def find_duplicates_by_hash(original_folder, additional_folder):
    """
    두 폴더에서 이미지를 비교하여 중복 이미지를 찾고 제거

    Args:
        original_folder (str): 원본 이미지 폴더 경로
        additional_folder (str): 추가 이미지 폴더 경로

    Steps:
        1. 원본 폴더에서 이미지 해시 값을 추출
        2. 추가 폴더에서 각 이미지의 해시 값을 계산하고, 원본 해시 값과 비교
        3. 중복된 이미지를 찾아 제거
        4. 중복되지 않은 추가 이미지는 JSON 형식으로 정보를 저장
    """
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
    '''
    original_folder = 원본 이미지 경로
    additional_folder = 비교 이미지 경로
    '''
    original_folder = "./original_images"
    additional_folder = "./new_images"

    find_duplicates_by_hash(original_folder, additional_folder)
