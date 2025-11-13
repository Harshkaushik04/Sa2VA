import argparse
import os
from transformers import BitsAndBytesConfig
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import cv2
import numpy as np
import spacy

# --------------------------
# Load NLP
# --------------------------
try:
    NLP_PARSER = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    NLP_PARSER = spacy.load("en_core_web_sm")


# --------------------------
# Extract nouns from text
# --------------------------
def extract_nouns(sentence):
    doc = NLP_PARSER(sentence)
    nouns = [chunk.root.text.lower() for chunk in doc.noun_chunks]
    return list(dict.fromkeys(nouns))  # unique order


# --------------------------
# Color mask overlay
# --------------------------
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


# --------------------------
# CLI arguments
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Sa2VA dual-mask segmentation (image or video) with relation mode")
    parser.add_argument("image_folder")
    parser.add_argument("--model_path", default="ByteDance/Sa2VA-B")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--work-dir", default="results")
    parser.add_argument("--select", type=int, default=-1)
    return parser.parse_args()


# ================================================================
#                         MAIN PROGRAM
# ================================================================
if __name__ == "__main__":
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)

    # Ask user image or video output
    print("\nChoose output type:")
    print("1 = Single Image Output")
    print("2 = Video Output")
    mode = input("Enter choice (1 or 2): ").strip()

    if mode not in ["1", "2"]:
        print("Invalid input. Choose 1 or 2.")
        exit()

    # --------------------------
    # Load model in 4-bit
    # --------------------------
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

    # --------------------------
    # Load frames
    # --------------------------
    image_paths = sorted([
        os.path.join(cfg.image_folder, f)
        for f in os.listdir(cfg.image_folder)
        if os.path.splitext(f)[1].lower() in {".jpg",".jpeg",".png",".bmp",".tiff"}
    ])
    vid_frames = [Image.open(p).convert("RGB") for p in image_paths]

    print(f"\nLoaded {len(vid_frames)} frames.\n")

    # --------------------------
    # Clean text & extract nouns
    # --------------------------
    clean_text = cfg.text.replace("<image>", "").strip()
    nouns = extract_nouns(clean_text)

    print("Extracted nouns:", nouns)

    if len(nouns) < 2:
        print("Need at least 2 objects. Found:", nouns)
        exit()

    obj1, obj2 = nouns[0], nouns[1]
    print("\nObject 1 =", obj1)
    print("Object 2 =", obj2, "\n")

    COLOR1 = (0,255,0)   # green
    COLOR2 = (0,0,255)   # red

    # --------------------------
    # Relation Mode ALWAYS ON:
    # "associated with" logic
    # --------------------------
    prompt1 = (
        f"<image> Segment ONLY the {obj1} associated with the {obj2}. "
        f"Do NOT segment any other {obj1}s. "
        f"Choose the {obj1} who is closest to or interacting with the {obj2}. "
        f"Highlight ONLY that specific {obj1}."
    )

    prompt2 = (
        f"<image> Segment ONLY the {obj2} associated with the {obj1}. "
        f"Do NOT segment any other {obj2}s. "
        f"Choose the {obj2} that is closest to or interacted with by the {obj1}. "
        f"Highlight ONLY that specific {obj2}."
    )

    print("Prompt1:", prompt1)
    print("Prompt2:", prompt2)

    # --------------------------
    # Run 1 — person (or obj1)
    # --------------------------
    if cfg.select > 0:
        frame = vid_frames[cfg.select - 1]
        res1 = model.predict_forward(
            image=frame,
            text=prompt1,
            tokenizer=tokenizer
        )
        frame_list = [cfg.select - 1]
    else:
        res1 = model.predict_forward(
            video=vid_frames,
            text=prompt1,
            tokenizer=tokenizer
        )
        frame_list = range(len(vid_frames))

    mask1_set = res1["prediction_masks"][0]

    # --------------------------
    # Run 2 — guitar (or obj2)
    # --------------------------
    if cfg.select > 0:
        frame = vid_frames[cfg.select - 1]
        res2 = model.predict_forward(
            image=frame,
            text=prompt2,
            tokenizer=tokenizer
        )
    else:
        res2 = model.predict_forward(
            video=vid_frames,
            text=prompt2,
            tokenizer=tokenizer
        )

    mask2_set = res2["prediction_masks"][0]

    # ======================================================
    #                 IMAGE OUTPUT MODE
    # ======================================================
    if mode == "1":
        idx = list(frame_list)[0]
        img = cv2.imread(image_paths[idx])

        img_green, m1 = apply_mask_color(img, mask1_set[idx], COLOR1)
        img_red,   m2 = apply_mask_color(img, mask2_set[idx], COLOR2)

        combined = img_green.copy()
        combined[m2 == 1] = img_red[m2 == 1]

        out_path = os.path.join(cfg.work_dir, "combined_output.png")
        cv2.imwrite(out_path, combined)

        print(f"\n✔ Single image saved at: {out_path}\n")
        exit()

    # ======================================================
    #                 VIDEO OUTPUT MODE
    # ======================================================
    if mode == "2":
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
            img_red,   m2 = apply_mask_color(img, mask2_set[idx], COLOR2)

            combined = img_green.copy()
            combined[m2 == 1] = img_red[m2 == 1]

            writer.write(combined)

        writer.release()
        print(f"\n✔ Video saved at: {video_path}\n")