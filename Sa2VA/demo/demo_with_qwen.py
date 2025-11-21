import argparse
import os
import torch
import cv2
import gc
import json
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Removed mmengine and matplotlib dependencies.
# We will use manual OpenCV blending for full control.

# Define a fixed palette of distinct BGR colors for objects (excluding black background)
# Format: (Blue, Green, Red)
BGR_PALETTE = [
    (0, 0, 255),   # ID 1: Red
    (0, 255, 0),   # ID 2: Green
    (255, 0, 0),   # ID 3: Blue
    (0, 255, 255), # ID 4: Yellow
    (255, 255, 0), # ID 5: Cyan
    (255, 0, 255), # ID 6: Magenta
    (0, 165, 255), # ID 7: Orange
    (128, 0, 128), # ID 8: Purple
]

def parse_args():
    parser = argparse.ArgumentParser(description='Full Pipeline: Sa2VA Description -> Qwen Reasoning -> Sa2VA Segmentation (Final Fixed Viz)')
    parser.add_argument('image_folder', help='Path to folder containing video frames')
    parser.add_argument('--query', type=str, required=True, help='The focus query (e.g., "Woman holding blue bottle")')
    
    # Models
    parser.add_argument('--sa2va_path', default="ByteDance/Sa2VA-1B", help='Path to Sa2VA model')
    parser.add_argument('--qwen_base', default="Qwen/Qwen2.5-3B-Instruct", help='Base Qwen model')
    parser.add_argument('--qwen_adapter', default="./qwen2.5-segmentation-finetune", help='Path to your fine-tuned adapter')
    
    # Outputs
    parser.add_argument('--work-dir', default="./results_final", help='Directory to save visualizations')
    
    # Optimization
    parser.add_argument('--max_frames', type=int, default=None, help='Limit frame count (e.g. 8). Default: All frames.')
    parser.add_argument('--resize_short_edge', type=int, default=None, help='Resize short edge (e.g. 448). Default: Full res.')
    
    return parser.parse_args()

# --- IMAGE UTILS ---
def get_image_files(folder_path):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    try:
        files = sorted([
            f for f in os.listdir(folder_path) 
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
        return [os.path.join(folder_path, f) for f in files]
    except Exception as e:
        print(f"Error reading folder {folder_path}: {e}")
        return []

def load_and_process_frames(image_paths, max_frames=None, resize_short=None):
    total_frames = len(image_paths)
    if max_frames is not None and total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        sampled_paths = [image_paths[i] for i in indices]
        print(f"Optimization: Sampling {max_frames} frames out of {total_frames}.")
    else:
        sampled_paths = image_paths
        
    loaded_frames = []
    for p in sampled_paths:
        try:
            img = Image.open(p).convert('RGB')
            if resize_short is not None:
                w, h = img.size
                scale = resize_short / min(w, h)
                if scale < 1.0:
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            loaded_frames.append(img)
        except Exception as e:
            print(f"Warning: Failed to load {p}: {e}")
    return loaded_frames, sampled_paths

# --- MODEL LOADING ---
def load_sa2va(model_path):
    print(f"Loading Sa2VA: {model_path}...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def load_qwen(base_path, adapter_path):
    print(f"Loading Qwen: {base_path} + {adapter_path}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer

# --- MAIN PIPELINE ---
def main():
    args = parse_args()
    
    # 1. Load Images
    all_paths = get_image_files(args.image_folder)
    if not all_paths:
        print("No images found.")
        return
    
    frames, processed_paths = load_and_process_frames(all_paths, args.max_frames, args.resize_short_edge)
    print(f"Loaded {len(frames)} frames.")

    # ==========================================
    # STAGE 1: Sa2VA Description Generation
    # ==========================================
    sa2va_model, sa2va_tokenizer = load_sa2va(args.sa2va_path)
    
    desc_prompt = "<image> Describe the video in detail. 1. Identify the main objects and describe their precise location (e.g., foreground, background, left, right, center). 2. Specify distinct visual attributes for each object, such as specific colors, clothing patterns, or unique physical actions. 3. If multiple similar objects are present, explicitly explain the visual feature or position that distinguishes one from the other."
    
    print("\n[Stage 1] Generating Video Description...")
    with torch.inference_mode():
        result = sa2va_model.predict_forward(
            video=frames,
            text=desc_prompt,
            tokenizer=sa2va_tokenizer
        )
    
    video_description = result['prediction']
    print(f"  > Description: {video_description}...")
    
    del sa2va_model
    gc.collect()
    torch.cuda.empty_cache()
    print("  > Sa2VA unloaded.")

    # ==========================================
    # STAGE 2: Qwen Reasoning (JSON Generation)
    # ==========================================
    qwen_model, qwen_tokenizer = load_qwen(args.qwen_base, args.qwen_adapter)
    
    print("\n[Stage 2] Generating Segmentation Targets...")
    qwen_prompt = f"""<|im_start|>system
You are a specialized AI for semantic segmentation. Output valid JSON only.<|im_end|>
<|im_start|>user
Focus Query: {args.query}
Image Description: {video_description}<|im_end|>
<|im_start|>assistant
"""
    inputs = qwen_tokenizer(qwen_prompt, return_tensors="pt").to("cuda")
    
    with torch.inference_mode():
        outputs = qwen_model.generate(**inputs, max_new_tokens=512)
    
    json_str = qwen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    try:
        start = json_str.find('{')
        end = json_str.rfind('}') + 1
        if start != -1 and end != -1:
            clean_json = json_str[start:end]
            segmentation_targets = json.loads(clean_json)
            print("  > Targets Generated:", json.dumps(segmentation_targets, indent=2))
        else:
            print("Failed to find JSON brackets in output.")
            print(json_str)
            return
    except Exception as e:
        print(f"Error parsing JSON from Qwen: {e}")
        print("Raw output:", json_str)
        return

    del qwen_model
    gc.collect()
    torch.cuda.empty_cache()
    print("  > Qwen unloaded.")

    # ==========================================
    # STAGE 3: Sa2VA Segmentation
    # ==========================================
    sa2va_model, sa2va_tokenizer = load_sa2va(args.sa2va_path)
    
    print("\n[Stage 3] Running Segmentation (Multicolor Labeling)...")
    os.makedirs(args.work_dir, exist_ok=True)

    # Get frame dimensions
    h, w = frames[0].size[1], frames[0].size[0]

    # Initialize semantic maps with zeros (background) for each frame
    # combined_seg_maps holds the Object ID (0, 1, 2...) for each pixel
    combined_seg_maps = {i: np.zeros((h, w), dtype=np.uint8) for i in range(len(frames))}
    
    # Start object ID counter at 1 (0 is reserved for background)
    object_id = 1
    
    for key, sentence in segmentation_targets.items():
        if not key.startswith("sentence"): continue
        
        print(f"  > Processing ID {object_id}: {sentence}")
        
        with torch.inference_mode():
            result = sa2va_model.predict_forward(
                video=frames,
                text=sentence,
                tokenizer=sa2va_tokenizer
            )
        
        # Sa2VA returns masks as float32 (sigmoid output 0.0-1.0)
        if 'prediction_masks' in result and len(result['prediction_masks']) > 0:
            frame_masks = result['prediction_masks'][0] 
            
            for i, mask in enumerate(frame_masks):
                if i >= len(frames): break
                
                # Threshold the mask at 0.5 to get boolean True/False
                # Where True, assign the current object_id to the map.
                # This overwrites any overlapping previous objects.
                combined_seg_maps[i] = np.where(mask > 0.5, object_id, combined_seg_maps[i])
        else:
            print(f"    Warning: No masks found for sentence ID {object_id}. Sa2VA may have failed to find this object.")
            
        # Increment ID for the next sentence/object
        object_id += 1

    # ==========================================
    # STAGE 4: Save Final Visualizations (FIXED)
    # ==========================================
    print("\n[Stage 4] Saving results (Manual OpenCV Blending)...")

    for i, frame_path in enumerate(processed_paths):
        seg_map = combined_seg_maps[i]
        # Convert PIL RGB frame to OpenCV BGR format
        img_bgr = np.array(frames[i])[:, :, ::-1].copy()

        # Check if any objects were detected in this frame
        unique_ids = np.unique(seg_map)
        has_objects = len(unique_ids) > 1 or (len(unique_ids) == 1 and unique_ids[0] != 0)

        if has_objects:
            # Create a copy of the image to paint solid colors on
            overlay_img = img_bgr.copy()

            for uid in unique_ids:
                if uid == 0: continue # Skip background

                # Get BGR color cyclically from fixed palette based on ID
                # Use (uid-1) so ID 1 gets index 0 (Red)
                color_bgr = BGR_PALETTE[(uid - 1) % len(BGR_PALETTE)]

                # Find pixels belonging to this object ID
                obj_mask = (seg_map == uid)

                # Paint solid color onto the overlay image wherever mask is True
                overlay_img[obj_mask] = color_bgr

            # Blend the original image and the painted overlay
            # alpha=0.6 means: 60% Original Image + 40% Painted Overlay
            # Background areas blend with themselves (no change). Object areas become transparently colored.
            alpha = 0.6
            cv2.addWeighted(img_bgr, alpha, overlay_img, 1 - alpha, 0, img_bgr)

        # Save final image (either unchanged original, or blended result)
        output_filename = f"seg_{os.path.basename(frame_path)}"
        output_path = os.path.join(args.work_dir, output_filename)
        cv2.imwrite(output_path, img_bgr)
            
    print(f"Done. Results saved to {args.work_dir}")

if __name__ == "__main__":
    main()