import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import torch
import cv2
import numpy as np

# =========================================================
# LOAD QWEN 2.5 FOR NOUN EXTRACTION
# =========================================================
print("Loading Qwen 2.5 noun extractor...")
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, device_map="cpu", trust_remote_code=True)

def qwen_extract_nouns(sentence):
    prompt = (
        "Extract the main object nouns from this sentence. "
        "Return only the nouns separated by commas. No explanation.\n\n"
        f"Sentence: {sentence}"
    )

    inputs = qwen_tokenizer(prompt, return_tensors="pt")
    outputs = qwen_model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False
    )
    result = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # clean
    text = result.split("Sentence:")[-1].strip().lower()
    text = text.replace(".", "").replace("and", ",")
    nouns = [x.strip() for x in text.split(",") if x.strip()]

    # Filtering clothing + colors
    clothing = {"shirt","tshirt","jeans","pant","pants","jacket","hoodie",
                "shorts","cap","hat","coat","sweater","skirt"}
    colors = {"red","blue","green","yellow","black","white","brown",
              "pink","orange","violet","purple","grey","gray"}

    nouns = [n for n in nouns if n not in clothing and n not in colors]

    # remove duplicates
    nouns = list(dict.fromkeys(nouns))

    # prioritise person + object
    if "person" in nouns and len(nouns) > 2:
        nouns = ["person"] + [n for n in nouns if n != "person"]

    return nouns[:2]  # max 2 nouns


# =========================================================
# MASK COLORING
# =========================================================
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


# =========================================================
# ARG PARSER
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Sa2VA + Qwen 2.5 segmentation")
    parser.add_argument("input_path")
    parser.add_argument("--model_path", default="ByteDance/Sa2VA-B")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--work-dir", default="results_qwen")
    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)

    print("\nLoading Sa2VA model...")
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

    # =========================================================
    # LOAD FRAMES
    # =========================================================
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

    print(f"\nLoaded {num_frames} frame(s).\n")

    # =========================================================
    # NOUNS
    # =========================================================
    clean_text = cfg.text.replace("<image>", "").strip()
    nouns = qwen_extract_nouns(clean_text)

    print("Extracted nouns:", nouns)

    if len(nouns) == 0:
        print("❌ No objects detected.")
        exit()

    if len(nouns) == 1:
        obj1 = nouns[0]
        obj2 = None
    else:
        obj1, obj2 = nouns[0], nouns[1]

    print(f"\nObject 1 = {obj1}")
    print(f"Object 2 = {obj2}\n")

    COLOR1 = (0,255,0)
    COLOR2 = (0,0,255)

    # =========================================================
    # PROMPTS
    # =========================================================
    if obj2:
        prompt1 = (
            f"<image> Segment ONLY the {obj1} associated with the {obj2}. "
            f"Do NOT segment any other {obj1}s."
        )

        prompt2 = (
            f"<image> Segment ONLY the {obj2} associated with the {obj1}. "
            f"Do NOT segment any other {obj2}s."
        )
    else:
        prompt1 = f"<image> Segment ONLY the {obj1}. Highlight it clearly."

    # =========================================================
    # IMAGE or VIDEO MODE
    # =========================================================
    is_video = num_frames > 1

    # -------------------- RUN 1 --------------------
    if is_video:
        res1 = model.predict_forward(video=vid_frames, text=prompt1, tokenizer=tokenizer)
        frame_range = range(num_frames)
    else:
        res1 = model.predict_forward(image=vid_frames[0], text=prompt1, tokenizer=tokenizer)
        frame_range = [0]

    mask1 = res1["prediction_masks"][0]

    # -------------------- RUN 2 --------------------
    if obj2:
        if is_video:
            res2 = model.predict_forward(video=vid_frames, text=prompt2, tokenizer=tokenizer)
        else:
            res2 = model.predict_forward(image=vid_frames[0], text=prompt2, tokenizer=tokenizer)
        mask2 = res2["prediction_masks"][0]

    # =========================================================
    # IMAGE OUTPUT
    # =========================================================
    if not is_video:
        idx = 0
        img = cv2.imread(image_paths[idx])

        img1, m1 = apply_mask_color(img, mask1[idx], COLOR1)

        if obj2:
            img2, m2 = apply_mask_color(img, mask2[idx], COLOR2)
            combined = img1.copy()
            combined[m2 == 1] = img2[m2 == 1]
        else:
            combined = img1

        out_path = os.path.join(cfg.work_dir, "output.png")
        cv2.imwrite(out_path, combined)
        print(f"\n✔ Saved image: {out_path}\n")
        exit()

    # =========================================================
    # VIDEO OUTPUT
    # =========================================================
    print("\nCreating output video...")

    h, w, _ = cv2.imread(image_paths[0]).shape
    out_vid = os.path.join(cfg.work_dir, "output_video.mp4")

    writer = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

    for idx in frame_range:
        img = cv2.imread(image_paths[idx])
        img1, m1 = apply_mask_color(img, mask1[idx], COLOR1)

        if obj2:
            img2, m2 = apply_mask_color(img, mask2[idx], COLOR2)
            combined = img1.copy()
            combined[m2 == 1] = img2[m2 == 1]
        else:
            combined = img1

        writer.write(combined)

    writer.release()
    print(f"\n✔ Output video saved: {out_vid}\n")