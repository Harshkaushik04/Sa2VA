import argparse
import os
import re
import json
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
import cv2
import numpy as np
import spacy

# -----------------------------
# Load spaCy
# -----------------------------
try:
    NLP = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    NLP = spacy.load("en_core_web_sm")

# -----------------------------
# Extract ALL nouns robustly
# -----------------------------
def extract_nouns(text):
    text = re.sub(r"<\s*image\s*>", "", text, flags=re.IGNORECASE)
    doc = NLP(text)
    # print("doc:",doc)
    nouns = []

    for chunk in doc.noun_chunks:
        nouns.append(chunk.root.text.lower())

    for tok in doc:
        if tok.pos_ in ("NOUN", "PROPN"):
            nouns.append(tok.text.lower())

    nouns = list(dict.fromkeys(nouns))
    nouns = [n for n in nouns if n != "image"]

    return nouns

# -----------------------------
# Apply color mask
# -----------------------------
def apply_color_mask(img, mask, color):
    m = (mask > 0.5).astype("uint8")
    overlay = img.copy()
    col = np.zeros_like(img)
    col[:] = color
    overlay[m == 1] = cv2.addWeighted(img[m == 1], 0.4, col[m == 1], 0.6, 0)
    return overlay

# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("image_folder")
    p.add_argument("--model_path", default="ByteDance/Sa2VA-B")
    p.add_argument("--text", required=True)
    p.add_argument("--work-dir", default="results_multi")
    p.add_argument("--select", type=int, default=-1)
    return p.parse_args()

# =============================
# Main
# =============================
if __name__ == "__main__":
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)

    print("\nChoose output type:\n1 = Image\n2 = Video")
    mode = input("Enter choice: ").strip()

    if mode not in ["1", "2"]:
        print("Invalid selection.")
        exit()

    print("\nLoading Sa2VA model...")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)

    image_paths = sorted([
        os.path.join(cfg.image_folder, f)
        for f in os.listdir(cfg.image_folder)
        if os.path.splitext(f)[1].lower() in {".jpg", ".png", ".jpeg", ".bmp", ".tiff"}
    ])

    frames = [Image.open(p).convert("RGB") for p in image_paths]
    N = len(frames)

    print(f"\nLoaded {N} frames.")

    # -----------------------------
    # Extract ALL nouns
    # -----------------------------
    nouns = extract_nouns(cfg.text)
    print("Extracted objects:", nouns)

    if len(nouns) == 0:
        print("No objects detected.")
        exit()

    # -----------------------------
    # Build prompts for EACH object
    # -----------------------------
    prompts = []
    for obj in nouns:
        p = f"<image> Segment ONLY the {obj}. Highlight ONLY the {obj}."
        prompts.append((obj, p))

    # -----------------------------
    # Run segmentation for each object
    # -----------------------------
    all_object_masks = []

    for obj, prompt in prompts:
        print(f"\nRunning segmentation for: {obj}")
        print(prompt)

        if cfg.select > 0:
            frame = frames[cfg.select - 1]
            res = model.predict_forward(image=frame, text=prompt, tokenizer=tokenizer)
            masks = res["prediction_masks"][0][:1]
            frame_indices = [cfg.select - 1]
        else:
            res = model.predict_forward(video=frames, text=prompt, tokenizer=tokenizer)
            masks = res["prediction_masks"][0]
            frame_indices = range(N)

        all_object_masks.append((obj, masks))

    # -----------------------------
    # IMAGE output
    # -----------------------------
    if mode == "1":
        idx = list(frame_indices)[0]
        img = cv2.imread(image_paths[idx])
        out = img.copy()

        base_colors = [
            (0,255,0), (0,0,255), (255,0,0),
            (255,255,0), (255,0,255), (0,255,255)
        ]

        for i, (obj, masks) in enumerate(all_object_masks):
            color = base_colors[i % len(base_colors)]
            out = apply_color_mask(out, masks[idx], color)

        save_path = os.path.join(cfg.work_dir, "multi_output.png")
        cv2.imwrite(save_path, out)

        print(f"\n✔ Saved: {save_path}")
        exit()

    # -----------------------------
    # VIDEO output
    # -----------------------------
    h, w, _ = cv2.imread(image_paths[0]).shape
    out_video = os.path.join(cfg.work_dir, "multi_output_video.mp4")
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

    base_colors = [
        (0,255,0), (0,0,255), (255,0,0),
        (255,255,0), (255,0,255), (0,255,255)
    ]

    for f in range(N):
        img = cv2.imread(image_paths[f])
        frame_out = img.copy()

        for i, (obj, masks) in enumerate(all_object_masks):
            color = base_colors[i % len(base_colors)]
            frame_out = apply_color_mask(frame_out, masks[f], color)

        writer.write(frame_out)

    writer.release()
    print(f"\n✔ Video saved: {out_video}")
