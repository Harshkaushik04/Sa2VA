import argparse
import os
import re
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
import cv2
import numpy as np
import spacy

# --------------------------------------------------------
# Load SpaCy
# --------------------------------------------------------
try:
    NLP = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    NLP = spacy.load("en_core_web_sm")


# --------------------------------------------------------
# Extract FULL noun phrases + adjective+noun
# --------------------------------------------------------
def extract_nouns(text):
    text = re.sub(r"<\s*image\s*>", "", text, flags=re.IGNORECASE)
    doc = NLP(text)

    found = set()

    # 1. Noun chunks (full phrase: "red bottle")
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        if phrase != "image" and len(phrase) > 1:
            found.add(phrase)

    # 2. adjective + noun
    for tok in doc:
        if tok.pos_ == "NOUN":
            adjs = [child.text.lower() for child in tok.children if child.pos_ == "ADJ"]
            if adjs:
                found.add(" ".join(adjs + [tok.text.lower()]))
            else:
                found.add(tok.text.lower())

    # cleanup duplicates
    final = []
    for f in found:
        if f not in final and f != "image":
            final.append(f)

    return final


# --------------------------------------------------------
# Relationship detector: e.g., man holding bottle
# --------------------------------------------------------
def detect_relationship(text):
    doc = NLP(text.lower())

    subject = None
    obj = None

    for tok in doc:
        if tok.dep_ in ("nsubj", "nsubjpass") and tok.pos_ in ("NOUN", "PROPN"):
            subject = tok.text.lower()

        if tok.dep_ in ("dobj", "pobj") and tok.pos_ in ("NOUN", "PROPN"):
            obj = tok.text.lower()

    if subject and obj:
        return subject, obj
    return None, None


# --------------------------------------------------------
# Build SMART prompts
# --------------------------------------------------------
def build_prompts(nouns, text):
    subject, obj = detect_relationship(text)
    prompts = []

    if subject and obj:
        # subject prompt
        prompts.append((
            subject,
            f"<image> Segment ONLY the {subject} who is associated with the {obj}. "
            f"Do NOT segment any other {subject}s. Choose the {subject} interacting with the {obj}. "
            f"Highlight ONLY that specific {subject}."
        ))

        # object prompt
        prompts.append((
            obj,
            f"<image> Segment ONLY the {obj} associated with the {subject}. "
            f"Do NOT segment any other {obj}s. Choose the {obj} being held or used by the {subject}. "
            f"Highlight ONLY that specific {obj}."
        ))

        # remove these nouns from normal list
        nouns = [n for n in nouns if subject not in n and obj not in n]

    # fallback: normal segmentation
    for n in nouns:
        prompts.append((n, f"<image> Segment ONLY the {n}. Do NOT segment any other objects."))

    return prompts


# --------------------------------------------------------
# Apply color mask
# --------------------------------------------------------
def apply_color_mask(img, mask, color):
    m = (mask > 0.5).astype("uint8")
    overlay = img.copy()
    color_layer = np.zeros_like(overlay)
    color_layer[:] = color
    overlay[m == 1] = cv2.addWeighted(overlay[m == 1], 0.4, color_layer[m == 1], 0.6, 0)
    return overlay


# --------------------------------------------------------
# Parse args
# --------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("image_folder")
    p.add_argument("--model_path", default="ByteDance/Sa2VA-B")
    p.add_argument("--text", required=True)
    p.add_argument("--work-dir", default="results_multi")
    p.add_argument("--select", type=int, default=-1)
    return p.parse_args()


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
if __name__ == "__main__":
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)

    print("\nChoose output type:\n1 = Image\n2 = Video")
    mode = input("Enter choice: ").strip()
    if mode not in ["1", "2"]:
        print("Invalid selection")
        exit()

    print("\nLoading Sa2VA...")
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

    # load frames
    image_paths = sorted([
        os.path.join(cfg.image_folder, f)
        for f in os.listdir(cfg.image_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])
    frames = [Image.open(p).convert("RGB") for p in image_paths]
    N = len(frames)
    print(f"Loaded {N} frame(s)")

    # extract nouns
    nouns = extract_nouns(cfg.text)
    print("Detected objects:", nouns)

    # build prompts with relationship logic
    prompts = build_prompts(nouns, cfg.text)
    print("\nGenerated prompts:")
    for obj, pp in prompts:
        print(f"  • {obj}: {pp}")

    # segmentation loop
    all_masks = []
    for obj, prompt in prompts:
        print(f"\nRunning segmentation for {obj}...")
        print(prompt)

        if cfg.select > 0:
            idx = cfg.select - 1
            res = model.predict_forward(image=frames[idx], text=prompt, tokenizer=tokenizer)
            masks = res["prediction_masks"][0][:1]
        else:
            res = model.predict_forward(video=frames, text=prompt, tokenizer=tokenizer)
            masks = res["prediction_masks"][0]

        all_masks.append(masks)

        # free GPU
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # image output
    if mode == "1":
        idx = cfg.select - 1 if cfg.select > 0 else 0
        img = cv2.imread(image_paths[idx])
        out = img.copy()

        COLORS = [
            (0,255,0), (0,0,255), (255,0,0),
            (255,255,0), (255,0,255), (0,255,255)
        ]

        for i, masks in enumerate(all_masks):
            out = apply_color_mask(out, masks[idx], COLORS[i % len(COLORS)])

        save_path = os.path.join(cfg.work_dir, "multimask_output.png")
        cv2.imwrite(save_path, out)
        print("Saved:", save_path)
        exit()

    # video output
    h, w, _ = cv2.imread(image_paths[0]).shape
    out_path = os.path.join(cfg.work_dir, "multimask_video.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

    COLORS = [
        (0,255,0), (0,0,255), (255,0,0),
        (255,255,0), (255,0,255), (0,255,255)
    ]

    for f in range(N):
        frame_out = cv2.imread(image_paths[f])

        for i, masks in enumerate(all_masks):
            frame_out = apply_color_mask(frame_out, masks[f], COLORS[i % len(COLORS)])

        writer.write(frame_out)

    writer.release()
    print("Saved:", out_path)