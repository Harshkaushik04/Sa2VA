import argparse
import os
import re
import json
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ---------------------------------------------
# Load Qwen model (CPU)
# ---------------------------------------------
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
print("Loading Qwen tokenizer & model (CPU) for prompt splitting...")
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, device_map="cpu", trust_remote_code=True)
qwen_model.eval()

# ---------------------------------------------
# Color palette
# ---------------------------------------------
BASE_COLORS = [
    (0,255,0), (0,0,255), (255,0,0),
    (255,255,0), (255,0,255), (0,255,255),
]

# ---------------------------------------------
# Apply mask color
# ---------------------------------------------
def apply_color_mask(img, mask, color):
    m = (mask > 0.5).astype("uint8")
    overlay = img.copy()
    col = np.zeros_like(img)
    col[:] = color
    overlay[m == 1] = cv2.addWeighted(img[m == 1], 0.35, col[m == 1], 0.65, 0)
    return overlay

# ---------------------------------------------
# Qwen-based multi-object segmentation
# ---------------------------------------------
def qwen_split(text, max_objects=6, max_tokens=256):
    if not text:
        return []

    system = (
        "You split a scene description into multiple segmentation prompts.\n"
        "Rules:\n"
        "1. Identify ALL visually distinct objects.\n"
        "2. Output ONLY JSON list of dicts: {\"object\": name, \"prompt\": prompt}.\n"
        "3. Prompt must ALWAYS be: '<image> Segment ONLY the X. Highlight ONLY the X.'\n"
        "4. No sentences, no explanations, JSON only."
    )

    full_prompt = system + "\nUser: " + text.strip() + "\n"

    inputs = qwen_tokenizer(full_prompt, return_tensors="pt")
    output = qwen_model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    raw = qwen_tokenizer.decode(output[0], skip_special_tokens=True)

    s, e = raw.find("["), raw.rfind("]")

    if s != -1 and e != -1:
        try:
            parsed = json.loads(raw[s:e+1])
            final = []
            for item in parsed[:max_objects]:
                obj = item.get("object", "").strip()
                prompt = item.get("prompt", "").strip()

                if not prompt.startswith("<image>"):
                    prompt = f"<image> Segment ONLY the {obj}. Highlight ONLY the {obj}."

                if obj:
                    final.append({"object": obj, "prompt": prompt})

            if final:
                return final
        except:
            pass

    # If parsing fails or no objects are found, raise an error.
    raise ValueError("Qwen failed to return valid JSON with objects; no fallback allowed.")

# ---------------------------------------------
# Argument parser
# ---------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("input_path")
    p.add_argument("--model_path", default="ByteDance/Sa2VA-1B")
    p.add_argument("--text", required=True)
    p.add_argument("--work-dir", default="results_qwen_multi")
    p.add_argument("--max-objects", type=int, default=6)
    return p.parse_args()

# ---------------------------------------------
# Main
# ---------------------------------------------
def main():
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)

    print("\nLoading Sa2VA model...")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    sa_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant,
        torch_dtype=torch.bfloat16
    )

    sa_tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)

    if os.path.isdir(cfg.input_path):
        images = sorted([
            os.path.join(cfg.input_path, f)
            for f in os.listdir(cfg.input_path)
            if os.path.splitext(f)[1].lower() in {".jpg", ".png", ".jpeg", ".bmp"}
        ])
    else:
        images = [cfg.input_path]

    frames = [Image.open(p).convert("RGB") for p in images]
    N = len(frames)
    print(f"Loaded {N} frames.")

    objects = qwen_split(cfg.text, max_objects=cfg.max_objects)
    print("Objects detected by Qwen:", [o["object"] for o in objects])

    all_masks = []
    for idx, obj in enumerate(objects):
        print(f"\nRunning segmentation for object: {obj['object']}")
        print(obj["prompt"])

        if N == 1:
            fake = [frames[0], frames[0]]
            res = sa_model.predict_forward(video=fake, text=obj["prompt"], tokenizer=sa_tokenizer)
            masks = res["prediction_masks"][0][:1]
        else:
            res = sa_model.predict_forward(video=frames, text=obj["prompt"], tokenizer=sa_tokenizer)
            masks = res["prediction_masks"][0]

        all_masks.append((obj["object"], masks))
        del res

    # ------------------ IMAGE OUTPUT ------------------
    if N == 1:
        img = cv2.imread(images[0])
        out = img.copy()

        for i, (obj, masks) in enumerate(all_masks):
            color = BASE_COLORS[i % len(BASE_COLORS)]
            out = apply_color_mask(out, masks[0], color)

        save = os.path.join(cfg.work_dir, "qwen_multi_output.png")
        cv2.imwrite(save, out)
        print("Saved image:", save)
        return

    # ------------------ VIDEO OUTPUT ------------------
    h, w, _ = cv2.imread(images[0]).shape
    out_vid = os.path.join(cfg.work_dir, "qwen_multi_video.mp4")
    writer = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

    for f in range(N):
        img = cv2.imread(images[f])
        out_frame = img.copy()

        for i, (obj, masks) in enumerate(all_masks):
            color = BASE_COLORS[i % len(BASE_COLORS)]
            out_frame = apply_color_mask(out_frame, masks[f], color)

        writer.write(out_frame)

    writer.release()
    print("Saved video:", out_vid)


if __name__ == "__main__":
    main()