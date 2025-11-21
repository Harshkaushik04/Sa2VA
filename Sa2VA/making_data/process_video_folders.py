import argparse
import os
import json
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Video Description with Sa2VA')
    parser.add_argument('image_folder', help='Root path containing numbered subfolders (1, 2, 3...)')
    parser.add_argument('--model_path', default="ByteDance/Sa2VA-1B", help='Path to the model')
    parser.add_argument('--output_file', default="output_descriptions.txt", help='Path for the output txt/json file')
    return parser.parse_args()

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    
    # Load model with specific settings from your demo.py
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
    return model, tokenizer

def get_images_from_folder(folder_path):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    try:
        image_files = sorted([
            f for f in os.listdir(folder_path) 
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
    except Exception as e:
        return []
    
    frames = []
    for filename in image_files:
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            frames.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
            
    return frames

def main():
    args = parse_args()
    
    # 1. Load Model (Once)
    model, tokenizer = load_model(args.model_path)
    
    # 2. Find and Sort Subfolders (1, 2, 3...)
    try:
        all_items = os.listdir(args.image_folder)
    except FileNotFoundError:
        print(f"Error: The folder '{args.image_folder}' does not exist.")
        return

    # Filter for directories and sort them numerically
    subfolders = []
    for item in all_items:
        full_path = os.path.join(args.image_folder, item)
        if os.path.isdir(full_path):
            subfolders.append(item)
    
    # Sort logic: Try to sort as integers, fall back to string sort if non-numeric
    subfolders.sort(key=lambda f: int(f) if f.isdigit() else f)
    
    print(f"Found {len(subfolders)} folders to process: {subfolders}")
    
    final_results = {}

    # 3. Process Each Folder
    prompt_text = "<image> Objectively analyze the visual scene for semantic segmentation. 1. Identify the main objects and describe their precise location (e.g., foreground, background, left, right, center). 2. Specify distinct visual attributes for each object, such as specific colors, clothing patterns, or unique physical actions. 3. If multiple similar objects (e.g., two dogs) are present, explicitly explain the visual feature or position that distinguishes one from the other."    
    for folder_name in subfolders:
        folder_path = os.path.join(args.image_folder, folder_name)
        print(f"Processing folder: {folder_name}...")
        
        frames = get_images_from_folder(folder_path)
        
        if not frames:
            print(f"  No valid images found in {folder_name}, skipping.")
            continue
            
        try:
            # Run Inference
            # FIX: Removed 'processor=processor' argument
            result = model.predict_forward(
                video=frames,
                text=prompt_text,
                tokenizer=tokenizer
            )
            
            description = result['prediction']
            print(f"  > Generated: {description[:50]}...") # Print preview
            
            # Store in dictionary
            final_results[folder_name] = {
                "FOCUS_QUERY": "",
                "IMAGE_DESCRIPTION": description
            }
            
        except Exception as e:
            print(f"  Error processing folder {folder_name}: {e}")

    # 4. Save to File
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        # Saving as JSON format inside the txt file as requested
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    main()