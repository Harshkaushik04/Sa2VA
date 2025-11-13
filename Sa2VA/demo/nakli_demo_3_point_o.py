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

def extract_nouns(sentence):
    doc = NLP_PARSER(sentence)
    nouns = [chunk.root.text.lower() for chunk in doc.noun_chunks]
    return list(dict.fromkeys(nouns))

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

def parse_args():
    parser = argparse.ArgumentParser(description="Sa2VA dual-mask segmentation (combined output)")
    parser.add_argument("image_folder")
    parser.add_argument("--model_path", default="ByteDance/Sa2VA-B")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--work-dir", default="results")
    parser.add_argument("--select", type=int, default=-1)
    return parser.parse_args()

if __name__ == "__main__":
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Load model (4bit)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
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

    print(f"\nLoaded {len(vid_frames)} frames\n")

    # Clean text
    clean_text = cfg.text.replace("<image>", "").strip()
    nouns = extract_nouns(clean_text)

    print("Extracted nouns:", nouns)

    if len(nouns) < 2:
        print("Need at least two objects. Found:", nouns)
        exit()

    obj1, obj2 = nouns[0], nouns[1]

    print("Object 1:", obj1)
    print("Object 2:", obj2)

    COLOR1 = (0,255,0)   # green
    COLOR2 = (0,0,255)   # red

    # ============================
    # Run 1 – segment obj1
    # ============================
    prompt1 = (
        f"<image> Segment ONLY the {obj1}. "
        f"Do NOT segment the {obj2}. "
        f"Highlight ONLY the {obj1}."
    )
    print("\nPrompt 1:", prompt1)

    if cfg.select > 0:
        fr = vid_frames[cfg.select-1]
        res1 = model.predict_forward(image=fr, text=prompt1, tokenizer=tokenizer)
        frame_list = [cfg.select - 1]
    else:
        res1 = model.predict_forward(video=vid_frames, text=prompt1, tokenizer=tokenizer)
        frame_list = range(len(vid_frames))

    mask1 = res1["prediction_masks"][0]

    # ============================
    # Run 2 – segment obj2
    # ============================
    prompt2 = (
        f"<image> Segment ONLY the {obj2}. "
        f"Do NOT segment the {obj1}. "
        f"The {obj2} is the object being held. Highlight ONLY the {obj2}."
    )
    print("\nPrompt 2:", prompt2)

    if cfg.select > 0:
        fr = vid_frames[cfg.select-1]
        res2 = model.predict_forward(image=fr, text=prompt2, tokenizer=tokenizer)
    else:
        res2 = model.predict_forward(video=vid_frames, text=prompt2, tokenizer=tokenizer)

    mask2 = res2["prediction_masks"][0]

    print("\nSaving combined outputs...\n")

    # ============================
    # COMBINE BOTH MASKS INTO ONE IMAGE
    # ============================
    for idx in frame_list:
        img = cv2.imread(image_paths[idx])

        # Apply both masks
        img_green, m1 = apply_mask_color(img, mask1[idx], COLOR1)
        img_red,   m2 = apply_mask_color(img, mask2[idx], COLOR2)

        # Combine masks: apply red over green image
        combined = img_green.copy()
        combined[m2 == 1] = img_red[m2 == 1]

        out_path = os.path.join(cfg.work_dir, f"combined_frame{idx}.png")
        cv2.imwrite(out_path, combined)

        print(f"Saved {out_path}")

    print("\n✔ Done! Combined mask image saved.\n")