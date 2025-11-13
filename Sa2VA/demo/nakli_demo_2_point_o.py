import argparse
import os
from transformers import BitsAndBytesConfig
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import cv2
import numpy as np

# NLP
import spacy
try:
    NLP_PARSER = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    NLP_PARSER = spacy.load("en_core_web_sm")

# Extract nouns
def extract_nouns(sentence):
    doc = NLP_PARSER(sentence)
    nouns = [chunk.root.text.lower() for chunk in doc.noun_chunks]
    return list(dict.fromkeys(nouns))  # unique + ordered

# Mask overlay
def overlay_mask(image, mask, color):
    mask = (mask > 0.5).astype("uint8")
    overlay = image.copy()
    colored = np.zeros_like(image)
    colored[:] = color
    overlay[mask == 1] = cv2.addWeighted(
        image[mask == 1], 0.4,
        colored[mask == 1], 0.6,
        0
    )
    return overlay

def parse_args():
    parser = argparse.ArgumentParser(description="Sa2VA dual-mask segmentation")
    parser.add_argument("image_folder")
    parser.add_argument("--model_path", default="ByteDance/Sa2VA-B")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--work-dir", default="results")
    parser.add_argument("--select", type=int, default=-1)
    return parser.parse_args()

if __name__ == "__main__":
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Load model
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)

    # Load frames
    image_paths = sorted([
        os.path.join(cfg.image_folder, f)
        for f in os.listdir(cfg.image_folder)
        if os.path.splitext(f)[1].lower() in {".jpg",".jpeg",".png",".bmp",".tiff"}
    ])
    vid_frames = [Image.open(p).convert("RGB") for p in image_paths]
    print(f"Loaded {len(vid_frames)} frames.")

    # -------------------------
    # FIX: Remove <image> token
    # -------------------------
    clean_text = cfg.text.replace("<image>", "").strip()
    nouns = extract_nouns(clean_text)

    print("\n====================")
    print("Extracted nouns:", nouns)
    print("====================\n")

    if len(nouns) < 2:
        print("Need at least 2 objects. Found:", nouns)
        exit()

    obj1, obj2 = nouns[0], nouns[1]
    print(f"Object 1 = {obj1}")
    print(f"Object 2 = {obj2}\n")

    # Colors
    COLOR1 = (0,255,0)   # green
    COLOR2 = (0,0,255)   # red

    # -----------------------------
    # FIRST RUN: person
    # -----------------------------
    prompt1 = (
        f"<image> Segment ONLY the {obj1}. "
        f"Do NOT segment the {obj2}. "
        f"Highlight ONLY the {obj1}. "
        f"Focus on the entire human figure."
    )
    print("Prompt 1:", prompt1)

    if cfg.select > 0:
        frame = vid_frames[cfg.select - 1]
        result1 = model.predict_forward(
            image=frame, text=prompt1, tokenizer=tokenizer
        )
        frames = [cfg.select - 1]
    else:
        result1 = model.predict_forward(
            video=vid_frames, text=prompt1, tokenizer=tokenizer
        )
        frames = range(len(vid_frames))

    mask_set1 = result1["prediction_masks"][0]

    # -----------------------------
    # SECOND RUN: guitar
    # -----------------------------
    prompt2 = (
        f"<image> Segment ONLY the {obj2}. "
        f"Do NOT segment the {obj1}. "
        f"The {obj2} is the object held by the person. "
        f"Highlight ONLY the {obj2}. "
        f"Focus on the instrument."
    )
    print("\nPrompt 2:", prompt2)

    if cfg.select > 0:
        frame = vid_frames[cfg.select - 1]
        result2 = model.predict_forward(
            image=frame, text=prompt2, tokenizer=tokenizer
        )
    else:
        result2 = model.predict_forward(
            video=vid_frames, text=prompt2, tokenizer=tokenizer
        )

    mask_set2 = result2["prediction_masks"][0]

    # -----------------------------
    # Save results
    # -----------------------------
    print("\nSaving outputs...\n")

    for idx in frames:
        img = cv2.imread(image_paths[idx])

        out1 = overlay_mask(img, mask_set1[idx], COLOR1)
        out2 = overlay_mask(img, mask_set2[idx], COLOR2)

        cv2.imwrite(os.path.join(cfg.work_dir, f"{obj1}_frame{idx}.png"), out1)
        cv2.imwrite(os.path.join(cfg.work_dir, f"{obj2}_frame{idx}.png"), out2)

    print(f"✔ Done. Masks saved in: {cfg.work_dir}")