import streamlit as st
import json
import os
from PIL import Image, ImageDraw, ImageFont
from streamlit_shortcuts import button

base_font_size = 30

'''
lang_image_folder (str) : 언어별 이미지 폴더 경로
lang_json (str) : 언어별 json 경로
lang_font (str) : 언어별 폰트 경로
'''

ch_image_folder = "ch_image_folder"
ch_json = "ch.json"
ch_font = "ch.ttf"

jp_image_folder = "jp_image_folder"
jp_json = "jp.json"
jp_font = "jp.ttf"

thai_image_folder = "thai_image_folder"
thai_json = "thai.json"
thai_font = "thai.ttf"

viet_image_folder = "viet_image_folder"
viet_json = "viet.json"
viet_font = "viet.ttf"

data_folders = {
    "Chinese Train": {"image_folder": ch_image_folder,
                      "json_path": ch_json,
                      "font_path" : ch_font,},
    "Japanese Train": {"image_folder": jp_image_folder,
                       "json_path": jp_json,
                       "font_path" : jp_font},
    "Thai Train": {"image_folder": thai_image_folder,
                   "json_path": thai_json,
                   "font_path" : thai_json},
    "Vietnamese Train": {"image_folder": viet_image_folder,
                         "json_path": viet_json,
                         "font_path" : viet_font}
    
}

st.title("UFO format OCR Visualization with PIL")

language_choice = st.selectbox("Select Language Train/Test", list(data_folders.keys()))
language_info = data_folders[language_choice]
json_path = language_info['json_path']
image_folder = language_info['image_folder']

@st.cache_data
def load_json_data(path):
    with open(path, 'r') as f:
        return json.load(f)

data = load_json_data(json_path)
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])

if "image_index" not in st.session_state:
    st.session_state.image_index = 0
if "bbox_index" not in st.session_state:
    st.session_state.bbox_index = 0

def update_image_index(offset):
    st.session_state.image_index = (st.session_state.image_index + offset) % len(image_files)
    st.session_state.bbox_index = 0


def update_bbox_index(offset, num_bboxes):
    st.session_state.bbox_index = (st.session_state.bbox_index + offset) % num_bboxes

col1, col2 = st.columns(2)
with col1:
    button("Next Image", "ArrowRight", update_image_index, args=(1,))
with col2:
    button("Previous Image", "ArrowLeft", update_image_index, args=(-1,))

current_image_file = image_files[st.session_state.image_index]
current_image_path = os.path.join(image_folder, current_image_file)
current_image_key = os.path.splitext(current_image_file)[0] if current_image_file not in data["images"] else current_image_file
current_image_info = data.get("images", {}).get(current_image_key, None)

st.subheader(f"Image: {current_image_file}")

display_mode = st.radio("Bounding Box Display Mode", ("All Boxes", "One Box at a Time"))

def display_image_with_boxes(image_path, image_info, display_mode):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    img_width, img_height = image.size
    font_size = int(min(img_width, img_height) * 0.03)
    
    font_path = language_info['font_path']
    font_prop = ImageFont.truetype(font_path, font_size)

    words = image_info.get("words", {})
    num_bboxes = len(words)

    if display_mode == "All Boxes":
        for word_id, word_info in words.items():
            draw_bbox(draw, word_info, font_prop)
    elif display_mode == "One Box at a Time" and num_bboxes > 0:
        word_ids = list(words.keys())
        word_info = words[word_ids[st.session_state.bbox_index]]
        draw_bbox(draw, word_info, font_prop)

        col3, col4 = st.columns(2)
        with col3:
            button("Previous Box", "ArrowDown", lambda: update_bbox_index(-1, len(current_image_info.get("words", {}))))
        with col4:
            button("Next Box", "ArrowUp", lambda: update_bbox_index(1, len(current_image_info.get("words", {}))))

    st.image(image, use_column_width=True)

def draw_bbox(draw, word_info, font_prop):
    transcription = word_info.get("transcription", "")
    points = [(x, y) for x, y in word_info.get("points", [])]
    class_label = word_info.get("class", "")

    if not transcription and not class_label:
        return

    x_min = min(p[0] for p in points)
    y_min = min(p[1] for p in points)
    x_max = max(p[0] for p in points)
    y_max = max(p[1] for p in points)

    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)

    display_text = f"{class_label}: {transcription}" if class_label else transcription
    
    try:
        text_bbox = font_prop.getbbox(display_text)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
    except TypeError:
        text_width, text_height = 0, 0

    padding_x = 0
    padding_y = 10

    text_x_min = x_min
    text_y_min = y_min - text_height - padding_y

    draw.rectangle([
        text_x_min - padding_x, 
        text_y_min - padding_y, 
        text_x_min + text_width + padding_x, 
        text_y_min + text_height + padding_y
    ], fill="red")

    draw.text((text_x_min, text_y_min+30), display_text, fill="white", font=font_prop, 
              stroke_width=2, stroke_fill="black", anchor="ls")

if current_image_info:
    display_image_with_boxes(current_image_path, current_image_info, display_mode)
else:
    st.warning(f"No annotation found for {current_image_file} in JSON.")
