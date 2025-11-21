import argparse
import os
import json
import torch
import gc
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Video Description with Sa2VA')
    parser.add_argument('image_folder', help='Root path containing numbered subfolders (1, 2, 3...)')
    parser.add_argument('--model_path', default="ByteDance/Sa2VA-1B", help='Path to the model')
    parser.add_argument('--output_file', default="output_descriptions.txt", help='Path for the output txt/json file')
    
    # OPTIONAL ARGUMENTS (Defaults are None, meaning full resolution/all frames)
    parser.add_argument('--max_frames', type=int, default=None, help='Optional: Limit number of frames from the start (e.g. 8 or 16). If not set, uses all frames.')
    parser.add_argument('--resize_short_edge', type=int, default=None, help='Optional: Resize shorter edge to this pixel count (e.g. 448). If not set, uses original resolution.')
    
    return parser.parse_args()

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def get_images_from_folder(folder_path, max_frames=None, resize_short=None):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    
    try:
        all_files = sorted([
            f for f in os.listdir(folder_path) 
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
    except Exception as e:
        print(f"Error reading folder {folder_path}: {e}")
        return []

    # OPTIONAL: Truncate frames only if argument is provided
    if max_frames is not None and len(all_files) > max_frames:
        print(f"  > Truncating to first {max_frames} frames (Original: {len(all_files)}).")
        all_files = all_files[:max_frames]
    
    frames = []
    for filename in all_files:
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            
            # OPTIONAL: Resize only if argument is provided
            if resize_short is not None:
                w, h = img.size
                scale = resize_short / min(w, h)
                if scale < 1.0: # Only downscale, never upscale
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            frames.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            
    return frames

def main():
    args = parse_args()
    
    # 1. Load Model
    model, tokenizer = load_model(args.model_path)
    
    # 2. Find and Sort Subfolders
    try:
        all_items = os.listdir(args.image_folder)
    except FileNotFoundError:
        print(f"Error: The folder '{args.image_folder}' does not exist.")
        return

    subfolders = []
    for item in all_items:
        full_path = os.path.join(args.image_folder, item)
        if os.path.isdir(full_path):
            subfolders.append(item)
    
    subfolders.sort(key=lambda f: int(f) if f.isdigit() else f)
    print(f"Found {len(subfolders)} folders to process.")
    
    final_results = {}
    
    # The optimized segmentation prompt
    # 3. Process Each Folder
    # UPDATED PROMPT: Removed the word "segmentation" to prevent [SEG] token output
    prompt_text = "<image> Describe the video in detail. 1. Identify the main objects and describe their precise location (e.g., foreground, background, left, right, center). 2. Specify distinct visual attributes for each object, such as specific colors, clothing patterns, or unique physical actions. 3. If multiple similar objects are present, explicitly explain the visual feature or position that distinguishes one from the other."
    # 3. Process Each Folder
    for folder_name in subfolders:
        folder_path = os.path.join(args.image_folder, folder_name)
        print(f"Processing folder: {folder_name}...")
        
        # Load images (will adhere to args for optional resizing/truncating)
        frames = get_images_from_folder(
            folder_path, 
            max_frames=args.max_frames, 
            resize_short=args.resize_short_edge
        )
        
        if not frames:
            print(f"  No valid images found in {folder_name}, skipping.")
            continue
            
        try:
            # Using inference_mode for efficiency
            with torch.inference_mode():
                result = model.predict_forward(
                    video=frames,
                    text=prompt_text,
                    tokenizer=tokenizer
                )
            
            description = result['prediction']
            print(f"  > Generated: {description}")
            
            final_results[folder_name] = {
                "FOCUS_QUERY": "",
                "IMAGE_DESCRIPTION": description
            }
            
        except Exception as e:
            print(f"  Error processing folder {folder_name}: {e}")
            if "out of memory" in str(e).lower():
                print("  ! GPU OOM Error. Re-run with --max_frames 8 or --resize_short_edge 448")
                torch.cuda.empty_cache()

        # Explicit cleanup to prevent VRAM creep over time
        del frames
        if 'result' in locals(): del result
        gc.collect()
        torch.cuda.empty_cache()

    # 4. Save to File
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    main()