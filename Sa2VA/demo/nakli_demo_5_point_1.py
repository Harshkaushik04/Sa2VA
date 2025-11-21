import argparse
import os
import json
import ast
import ollama
import requests  # For the VRAM fix
import time      # For the VRAM fix
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import torch
import cv2
import numpy as np

# -----------------------------
# Helper Functions
# -----------------------------

def apply_color_mask(img, mask, color):
    """Applies a colored mask overlay to an image."""
    m = (mask > 0.5).astype("uint8")
    overlay = img.copy()
    col = np.zeros_like(img)
    col[:] = color
    overlay[m == 1] = cv2.addWeighted(img[m == 1], 0.4, col[m == 1], 0.6, 0)
    return overlay

def load_sa2va_model(model_path):
    """Loads the Sa2VA model and tokenizer into VRAM."""
    print("\n--- Loading Sa2VA model into VRAM... ---")
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("--- Sa2VA model loaded. ---")
    return model, tokenizer

def unload_sa2va_model(model, tokenizer):
    """Removes the Sa2VA model and tokenizer from VRAM."""
    print("\n--- Unloading Sa2VA model from VRAM... ---")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    print("--- VRAM cleared. ---")

def robust_json_find(text: str) -> str:
    """Finds the first JSON object in a string."""
    start_index = text.find('{')
    end_index = text.rfind('}')
    if start_index == -1 or end_index == -1:
        raise ValueError("No JSON object '{}' found in the model's output.")
    return text[start_index : end_index + 1]

# -----------------------------
# Args
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Multi-object segmentation using Sa2VA and Ollama.")
    p.add_argument("image_folder", help="Path to a folder of sequential image frames.")
    p.add_argument("--model_path", default="ByteDance/Sa2VA-1B", help="Path or name of the Sa2VA model.")
    p.add_argument("--text", required=True, help="The input query to parse (e.g., 'woman holding bottle' or 'man and woman').")
    p.add_argument("--work-dir", default="results_multi", help="Directory to save output files.")
    p.add_argument("--select", type=int, default=-1, help="Select a single frame to process by its 1-based index.")
    return p.parse_args()

# =============================
# Main Pipeline
# =============================
if __name__ == "__main__":
    cfg = parse_args()
    os.makedirs(cfg.work_dir, exist_ok=True)

    print("\nChoose output type:\n1 = Image (from first or selected frame)\n2 = Video")
    mode = input("Enter choice: ").strip()

    if mode not in ["1", "2"]:
        print("Invalid selection.")
        exit()

    # -----------------------------
    # Load Frames (Low memory usage)
    # -----------------------------
    image_paths = sorted([
        os.path.join(cfg.image_folder, f)
        for f in os.listdir(cfg.image_folder)
        if os.path.splitext(f)[1].lower() in {".jpg", ".png", ".jpeg", ".bmp", ".tiff"}
    ])

    frames = [Image.open(p).convert("RGB") for p in image_paths]
    N = len(frames)
    print(f"\nLoaded {N} frames.")

    # -----------------------------
    # STEP 1: Get Image Description
    # (Sa2VA is loaded and unloaded)
    # -----------------------------
    model, tokenizer = load_sa2va_model(cfg.model_path)
    
    print("\n--- Pipeline Step 1: Generating image description ---")
    desc_prompt = "<image> describe image"
    
    if cfg.select > 0:
        frame_for_desc = frames[cfg.select - 1]
    else:
        frame_for_desc = frames[0] # Use first frame for video description

    try:
        desc_res = model.predict_forward(image=frame_for_desc, text=desc_prompt, tokenizer=tokenizer)
        image_description = desc_res['prediction'].strip()
    except Exception as e:
        print(f"Error during description generation: {e}")
        unload_sa2va_model(model, tokenizer)
        exit()
        
    print(f"Image Description: {image_description}")
    
    # UNLOAD Sa2VA to make room for Ollama
    unload_sa2va_model(model, tokenizer)

    # -----------------------------
    # STEP 2: Get Segmentation Prompts (Single Ollama Call)
    # (Sa2VA is NOT in memory)
    # -----------------------------
    print(f"\n--- Pipeline Step 2: Generating segmentation sentences... ---")
    
    # **** THIS IS THE NEW, FINAL "MEGA-PROMPT" ****
    # It includes examples for all 3 of your query types.
    segmenter_system_prompt = """You are an AI data extractor. Your task is to perform a specific analysis and return *only* a JSON object.

**TASK:**
1.  Read the `IMAGE_DESCRIPTION` and the `FOCUS_QUERY`.
2.  Your goal is to find *all* distinct objects in the `IMAGE_DESCRIPTION` that **fully match** the `FOCUS_QUERY`.
3.  **CRITICAL RULE:** If the `FOCUS_QUERY` describes an interaction (e.g., "A holding B", "A with B"), you MUST generate a separate sentence for "A" and a separate sentence for "B". Do NOT fuse them into one.
4.  For *each* matching object, generate a comprehensive sentence describing it.
5.  Each sentence **must** start with the exact prefix: `<image> Segment this object which...`
6.  Format this output *only* as a JSON object: {sentence1: "...", sentence2: "..."}

**CONSTRAINTS:**
* Your output MUST be *only* the JSON object. No other text.
* Your response MUST begin with { and end with }.
* If no objects match, return an empty JSON object `{}`.

**EXAMPLE 1 (Filtering):**
* `FOCUS_QUERY`: "person with blue shirt"
* `IMAGE_DESCRIPTION`: "A man in a blue shirt... A woman in a red shirt..."
* `EXPECTED OUTPUT`: {"sentence1": "<image> Segment this object which is a man in a blue shirt"}

**EXAMPLE 2 (Interaction Rule):**
* `FOCUS_QUERY`: "woman holding bottle"
* `IMAGE_DESCRIPTION`: "...The woman is wearing a red shirt and holding a blue thermos... the man is wearing a blue shirt and holding a red thermos..."
* `EXPECTED OUTPUT`: {"sentence1": "<image> Segment this object which is the woman wearing a red shirt", "sentence2": "<image> Segment this object which is the blue thermos held by the woman"}

**EXAMPLE 3 (List of Objects):**
* `FOCUS_QUERY`: "man and woman"
* `IMAGE_DESCRIPTION`: "...The woman is wearing a red shirt... the man is wearing a blue shirt..."
* `EXPECTED OUTPUT`: {"sentence1": "<image> Segment this object which is the man wearing a blue shirt", "sentence2": "<image> Segment this object which is the woman wearing a red shirt"}
"""
    
    clean_text = cfg.text.replace("<image>", "").strip()
    segmenter_user_prompt = f"""IMAGE_DESCRIPTION: "{image_description}"
<|im_end|>
FOCUS_QUERY: "{clean_text}"
"""
    
    raw_json_output = ""
    segment_json_string = ""
    try:
        final_response = ollama.chat(
            model='llama3:8b',
            messages=[
                {'role': 'system', 'content': segmenter_system_prompt},
                {'role': 'user', 'content': segmenter_user_prompt},
            ],
            options={'keep_alive': '0m'} # Tell Ollama to unload after
        )
        
        # --- ROBUST PARSING LOGIC ---
        raw_json_output = final_response['message']['content']
        segment_json_string = robust_json_find(raw_json_output)
        segment_prompts_dict = json.loads(segment_json_string)
        # --- END OF ROBUST LOGIC ---
        
        print(f"\n--- FINAL JSON OUTPUT ---")
        print(segment_json_string)
        
    except Exception as e:
        print(f"\n--- ERROR: Could not parse the final JSON from Ollama. ---")
        print(f"Model returned: {raw_json_output}")
        print(f"Error: {e}")
        exit()

    # -----------------------------
    # **** VRAM FIX (Using your solution) ****
    # -----------------------------
    try:
        print("\n--- Forcing Ollama to unload llama3:8b from VRAM... ---")
        requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3:8b",
                "keep_alive": 0, # 0 seconds
            },
        )
        print("--- Ollama model unload command sent. Waiting 2 seconds... ---")
        time.sleep(2) # Give the server a moment to process the unload
    except Exception as e:
        print(f"Warning: Could not send unload command to Ollama. {e}")
    # -----------------------------
    # END OF FIX
    # -----------------------------


    # -----------------------------
    # STEP 3: Run Segmentation (RE-LOAD Sa2VA)
    # (Ollama is done, Sa2VA is loaded)
    # -----------------------------
    
    # This call should now succeed
    model, tokenizer = load_sa2va_model(cfg.model_path)
    
    all_object_masks = []
    base_colors = [
        (0,255,0), (0,0,255), (255,0,0),
        (255,255,0), (255,0,255), (0,255,255)
    ]
    
    prompts_for_loop = list(segment_prompts_dict.items())

    for i, (key, prompt) in enumerate(prompts_for_loop):
        # Create a simple label
        obj_name = f"Object {i+1}"
        
        print(f"\n--- Running segmentation for: {obj_name} ---")
        print(prompt)

        # Make sure the prompt has the <image> token (our prompt logic should do this)
        if "<image>" not in prompt:
            print("!!! WARNING: <image> token missing from prompt. Segmentation may fail. !!!")
            # We'll add it just in case, though the system prompt should handle this
            prompt = f"<image> {prompt}"
            
        try:
            if cfg.select > 0 or N == 1:
                frame = frames[cfg.select - 1 if cfg.select > 0 else 0]
                res = model.predict_forward(image=frame, text=prompt, tokenizer=tokenizer)
                masks = res["prediction_masks"][0]
                frame_indices = [cfg.select - 1 if cfg.select > 0 else 0]
            else:
                res = model.predict_forward(video=frames, text=prompt, tokenizer=tokenizer)
                masks = res["prediction_masks"][0]
                frame_indices = range(N)

            all_object_masks.append((obj_name, masks))
        
        except Exception as e:
            print(f"!!! WARNING: Segmentation failed for '{obj_name}'. Skipping. !!!")
            print(f"This often happens if the object isn't found or prompt is malformed.")
            print(f"Error: {e}")
            continue
            
    # UNLOAD Sa2VA for the final time
    unload_sa2va_model(model, tokenizer)

    # -----------------------------
    # STEP 4: Save Output (IMAGE)
    # (No models are in memory)
    # -----------------------------
    if mode == "1":
        idx = list(frame_indices)[0]
        img = cv2.imread(image_paths[idx])
        out = img.copy()

        print(f"\nApplying {len(all_object_masks)} masks to image...")
        
        for i, (obj, masks) in enumerate(all_object_masks):
            color = base_colors[i % len(base_colors)]
            out = apply_color_mask(out, masks[0], color) 

        save_path = os.path.join(cfg.work_dir, "multi_output.png")
        cv2.imwrite(save_path, out)
        print(f"\n✔ Saved: {save_path}")

    # -----------------------------
    # STEP 4: Save Output (VIDEO)
    # (No models are in memory)
    # -----------------------------
    if mode == "2":
        h, w, _ = cv2.imread(image_paths[0]).shape
        out_video = os.path.join(cfg.work_dir, "multi_output_video.mp4")
        writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

        print(f"\nApplying {len(all_object_masks)} masks to video...")

        for f in range(N):
            img = cv2.imread(image_paths[f])
            frame_out = img.copy()

            for i, (obj, masks) in enumerate(all_object_masks):
                color = base_colors[i % len(base_colors)]
                frame_out = apply_color_mask(frame_out, masks[f], color)

            writer.write(frame_out)
            
        writer.release()
        print(f"\n✔ Video saved: {out_video}")

    print("\nPipeline complete.")