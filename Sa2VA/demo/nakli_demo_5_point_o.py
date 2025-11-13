import argparse
import os
from transformers import BitsAndBytesConfig
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import cv2
import numpy as np
import spacy

# ========================================================
# LOAD NLP MODEL
# ========================================================
try:
    NLP_PARSER = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    NLP_PARSER = spacy.load("en_core_web_sm")


# ========================================================
# EXTRACT NOUNS (ROBUST)
# ========================================================
def extract_nouns(sentence):
    doc = NLP_PARSER(sentence)
    nouns = [chunk.root.text.lower() for chunk in doc.noun_chunks]

    # Clothing
    clothing = {
        "shirt","tshirt","jeans","pant","pants","jacket","hoodie",
        "shorts","cap","hat","coat","sweater","skirt"
    }

    # Colors
    colors = {
        "red","blue","green","yellow","black","white","brown",
        "pink","orange","violet","purple","grey","gray"
    }

    # Remove useless nouns
    filtered = [n for n in nouns if n not in clothing and n not in colors]

    # Deduplicate
    filtered = list(dict.fromkeys(filtered))

    # CASE 1: No objects
    if len(filtered) == 0:
        return []

    # CASE 2: Only one object
    if len(filtered) == 1:
        return filtered

    # CASE 3: More than 2 nouns → prefer "person" + other object
    if "person" in filtered and len(filtered) > 2:
        filtered = ["person"] + [n for n in filtered if n != "person"]

    # Keep only first 2
    return filtered[:2]


# ========================================================
# COLOR MASK OVERLAY
# ========================================================
def apply_mask_color(img, mask, color):
    mask = (mask > 0.5).astype("uint8")
    overlay = img.copy()
    col = np.zeros_like(img)
    col[:] = color
    overlay[mask == 1] = cv2.addWeighted(
        img[mask == 1], 0.4,
        col[mask == 1], 0.6,
        0
    )
    return overlay, mask


# ========================================================
# ARG PARSER
# ========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Sa2VA auto dual-mask segmentation")
    parser.add_argument("input_path")
    parser.add_argument("--model_path", default="ByteDance/Sa2VA-B")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--work-dir", default="results")
    return parser.parse_args()


# ========================================================
# MAIN
# ========================================================
if __name__ == "__main__":
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)

    print("\nLoading model...")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)

    # ========================================================
    # LOAD INPUT FRAMES
    # ========================================================
    if os.path.isdir(cfg.input_path):
        files = sorted(os.listdir(cfg.input_path))
        image_paths = [
            os.path.join(cfg.input_path, f)
            for f in files
            if os.path.splitext(f)[1].lower() in {".jpg",".jpeg",".png",".bmp",".tiff"}
        ]
    else:
        image_paths = [cfg.input_path]

    vid_frames = [Image.open(p).convert("RGB") for p in image_paths]
    num_frames = len(vid_frames)

    print(f"\nLoaded {num_frames} frames.\n")

    # ========================================================
    # NOUN EXTRACTION
    # ========================================================
    clean_text = cfg.text.replace("<image>", "").strip()
    nouns = extract_nouns(clean_text)

    print("Extracted nouns:", nouns)

    if len(nouns) == 0:
        print("❌ No valid objects found from the sentence.")
        exit()

    # CASE: 1 object only → give single mask
    if len(nouns) == 1:
        obj1 = nouns[0]
        obj2 = None
    else:
        obj1, obj2 = nouns[0], nouns[1]

    print("\nObject 1 =", obj1)
    print("Object 2 =", obj2, "\n")

    COLOR1 = (0,255,0)   # green
    COLOR2 = (0,0,255)   # red

    # ========================================================
    # BUILD PROMPTS
    # ========================================================
    # relation-aware prompt for obj1
    if obj2:
        prompt1 = (
            f"<image> Segment ONLY the {obj1} associated with the {obj2}. "
            f"Do NOT segment any other {obj1}s. "
            f"Select the {obj1} closest to or interacting with the {obj2}."
        )
    else:
        prompt1 = f"<image> Segment ONLY the {obj1}. Highlight the {obj1} clearly."

    # relation-aware prompt for obj2 (if exists)
    if obj2:
        prompt2 = (
            f"<image> Segment ONLY the {obj2} associated with the {obj1}. "
            f"Do NOT segment any other {obj2}s. "
            f"Select the {obj2} closest to or interacted with by the {obj1}."
        )

    # ========================================================
    # DETERMINE MODE AUTOMATICALLY
    # ========================================================
    is_video = num_frames > 1

    # ========================================================
    # RUN 1
    # ========================================================
    if is_video:
        res1 = model.predict_forward(
            video=vid_frames,
            text=prompt1,
            tokenizer=tokenizer
        )
        frame_list = range(num_frames)
    else:
        res1 = model.predict_forward(
            image=vid_frames[0],
            text=prompt1,
            tokenizer=tokenizer
        )
        frame_list = [0]

    mask1_set = res1["prediction_masks"][0]

    # ========================================================
    # RUN 2 (IF SECOND OBJECT EXISTS)
    # ========================================================
    if obj2:
        if is_video:
            res2 = model.predict_forward(
                video=vid_frames,
                text=prompt2,
                tokenizer=tokenizer
            )
        else:
            res2 = model.predict_forward(
                image=vid_frames[0],
                text=prompt2,
                tokenizer=tokenizer
            )

        mask2_set = res2["prediction_masks"][0]

    # ========================================================
    # IMAGE OUTPUT MODE
    # ========================================================
    if not is_video:
        idx = 0
        img = cv2.imread(image_paths[idx])

        img_green, m1 = apply_mask_color(img, mask1_set[idx], COLOR1)

        if obj2:
            img_red,   m2 = apply_mask_color(img, mask2_set[idx], COLOR2)
            combined = img_green.copy()
            combined[m2 == 1] = img_red[m2 == 1]
        else:
            combined = img_green

        out_path = os.path.join(cfg.work_dir, "combined_output.png")
        cv2.imwrite(out_path, combined)

        print(f"\n✔ Single combined image saved: {out_path}\n")
        exit()

    # ========================================================
    # VIDEO OUTPUT MODE
    # ========================================================
    print("\nGenerating video...")

    h, w, _ = cv2.imread(image_paths[0]).shape
    video_path = os.path.join(cfg.work_dir, "combined_output_video.mp4")
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (w, h)
    )

    for idx in frame_list:
        img = cv2.imread(image_paths[idx])
        img_green, m1 = apply_mask_color(img, mask1_set[idx], COLOR1)

        if obj2:
            img_red, m2 = apply_mask_color(img, mask2_set[idx], COLOR2)
            combined = img_green.copy()
            combined[m2 == 1] = img_red[m2 == 1]
        else:
            combined = img_green

        writer.write(combined)

    writer.release()
    print(f"✔ Video saved: {video_path}\n")