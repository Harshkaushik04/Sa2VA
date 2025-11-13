import argparse
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import torch
import cv2
import numpy as np

# ------------------------------
# Qwen model
# ------------------------------
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
print("Loading Qwen tokenizer & model (CPU) for prompt splitting...")
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, device_map="cpu", trust_remote_code=True)
qwen_model.eval()

# ------------------------------
# Colors
# ------------------------------
BASE_COLORS = [
    (0, 255, 0),   # green
    (0, 0, 255),   # red
    (255, 0, 0),   # blue
    (255, 255, 0), # cyan
    (255, 0, 255)  # magenta
]

# ------------------------------
# Mask coloring
# ------------------------------
def apply_color_mask(img, mask, color):
    m = (mask > 0.5).astype("uint8")
    overlay = img.copy()
    color_arr = np.zeros_like(img)
    color_arr[:] = color
    overlay[m == 1] = cv2.addWeighted(img[m == 1], 0.3, color_arr[m == 1], 0.7, 0)
    return overlay

# ------------------------------
# Qwen prompt splitter
# ------------------------------
def qwen_split_to_prompts(text, max_objects=5, max_tokens=256):
    if not text:
        return []

    system = (
        "You split a scene description into multiple segmentation prompts.\n"
        "Rules:\n"
        "1) Identify visually distinct objects.\n"
        "2) Output JSON list with keys 'object' and 'prompt'.\n"
        "3) Prompt must be '<image> Segment ONLY the X. Highlight ONLY the X.'\n"
        "4) Never output natural sentences.\n"
        "5) Output only JSON."
    )

    full_prompt = system + "\nUser: " + text.strip() + "\n"

    inputs = qwen_tokenizer(full_prompt, return_tensors="pt")
    outputs = qwen_model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False
    )

    raw = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    s = raw.find("[")
    e = raw.rfind("]")

    if s != -1 and e != -1:
        try:
            parsed = json.loads(raw[s:e+1])
            out = []

            for item in parsed[:max_objects]:
                obj = item.get("object", "").strip()
                prompt = item.get("prompt", "").strip()

                if not prompt.startswith("<image>"):
                    prompt = f"<image> Segment ONLY the {obj}. Highlight ONLY the {obj}."

                if obj:
                    out.append({"object": obj, "prompt": prompt})

            if out:
                return out
        except:
            pass

    clean = text.replace("<image>", "").strip()
    fallback = f"<image> Segment ONLY the {clean}. Highlight ONLY the {clean}."
    return [{"object": clean, "prompt": fallback}]

# ------------------------------
# Arg parser
# ------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_path")
    ap.add_argument("--model_path", default="ByteDance/Sa2VA-1B")
    ap.add_argument("--text", required=True)
    ap.add_argument("--work-dir", default="results_new")
    ap.add_argument("--max-objects", type=int, default=5)
    return ap.parse_args()

# ------------------------------
# Main script
# ------------------------------
def main():
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)

    print("Loading Sa2VA model...")
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

    sa_model.eval()
    torch.set_grad_enabled(False)

    sa_tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path, trust_remote_code=True
    )

    if os.path.isdir(cfg.input_path):
        files = sorted(os.listdir(cfg.input_path))
        image_paths = [
            os.path.join(cfg.input_path, f)
            for f in files
            if os.path.splitext(f)[1].lower() in {".jpg", ".png", ".jpeg", ".bmp"}
        ]
    else:
        image_paths = [cfg.input_path]

    frames = [Image.open(p).convert("RGB") for p in image_paths]
    N = len(frames)

    print(f"Loaded {N} frame(s).")

    objects = qwen_split_to_prompts(cfg.text, max_objects=cfg.max_objects)
    print("Qwen objects:", [o["object"] for o in objects])

    all_masks = []
    for idx, obj in enumerate(objects):
        prompt = obj["prompt"]

        print(f"\nSegmentation {idx}: {obj['object']}")
        print(prompt)

        if N == 1:
            fake_video = [frames[0], frames[0]]
            res = sa_model.predict_forward(
                video=fake_video, text=prompt, tokenizer=sa_tokenizer
            )
            masks = res["prediction_masks"][0][:1]
        else:
            res = sa_model.predict_forward(
                video=frames, text=prompt, tokenizer=sa_tokenizer
            )
            masks = res["prediction_masks"][0]

        all_masks.append(masks)

        del res

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except:
                pass

    if N == 1:
        img = cv2.imread(image_paths[0])
        out = img.copy()

        for i, masks in enumerate(all_masks):
            out = apply_color_mask(out, masks[0], BASE_COLORS[i % len(BASE_COLORS)])

        save_path = os.path.join(cfg.work_dir, "combined_output.png")
        cv2.imwrite(save_path, out)

        print("Saved:", save_path)
        return

    h, w, _ = cv2.imread(image_paths[0]).shape

    out_video = os.path.join(cfg.work_dir, "combined_video.mp4")
    writer = cv2.VideoWriter(
        out_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (w, h)
    )

    for f in range(N):
        img = cv2.imread(image_paths[f])
        frame_out = img.copy()

        for i, masks in enumerate(all_masks):
            frame_out = apply_color_mask(
                frame_out,
                masks[f],
                BASE_COLORS[i % len(BASE_COLORS)]
            )

        writer.write(frame_out)

    writer.release()
    print("Saved:", out_video)

if __name__ == "__main__":
    main()