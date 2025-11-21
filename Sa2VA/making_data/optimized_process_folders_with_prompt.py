import argparse
import os
import json
import torch
import gc
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- DATA MAPPING ---
# Mapping folder IDs to their Focus Queries based on your provided data
QUERY_DATA = {
    "1": "Man riding horse",
    "2": "black car following white car",
    "3": "left aeroplane moving forward",
    "4": "two bears colliding heads with each other",
    "5": "Monkey going from door and coming back",
    "6": "Man feeding pandas",
    "7": "dog fighting monkeys",
    "8": "blue birds inside cage",
    "9": "tortoise swimming inside water",
    "11": "kid playing with dog", 
    "12": "monkey playing with rock",
    "13": "tortoise inside water",
    "14": "yellow truck and red truck",
    "15": "Man feeding sheeps",
    "16": "Rabits eating grass",
     "17": "bears fighting", 
    "18": "white dog fighting with black one",
    "19": "Tortoise coming from above",
    "20": "Elephant putting trunk on another elephant",
    "21": "person walking down his bicycle",
    "22": "child feeding white rabbit",
    "23": "hand raising alligator from one position to another",
    "24": "child riding bicyle",
    "25": "dogs fighting", 
    "27": "lizard on the left hand",
    "28": "Dogs fighting",
    "29": "monkeys fighting",
    "30": "airplane faced right",
    "31": "bear with its childs",
    "32": "birds inside green cage",
     "33": "fishes swimming", 
     "34": "goats walking", 
    "35": "monkeys with red shirt",
    "36": "tigers drinking water",
    "37": "cat with yellow device biting another car",
    "38": "horse running around man with red shirt",
    "39": "cats playing",
    "40": "tied up brown cow",
    "41": "elephant in the background",
    "42": "tortoise trying to go out of water",
    "43": "yawks fighting each other",
    "44": "pandas eating food",
     "45": "cows walking", 
    "47": "bulls fighting",
    "48": "rabits eating grass",
    "49": "tigers fighting",
    "50": "black horse"
}

def parse_args():
    parser = argparse.ArgumentParser(description='Batch Video Description with Sa2VA')
    parser.add_argument('image_folder', help='Root path containing numbered subfolders (1, 2, 3...)')
    parser.add_argument('--model_path', default="ByteDance/Sa2VA-1B", help='Path to the model')
    parser.add_argument('--output_file', default="output_descriptions.txt", help='Path for the output txt/json file')
    
    # OPTIMIZATIONS
    parser.add_argument('--max_frames', type=int, default=None, help='Limit frames (e.g. 8). Default: All frames.')
    parser.add_argument('--resize_short_edge', type=int, default=None, help='Resize short edge (e.g. 448). Default: Full res.')
    
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

    if max_frames is not None and len(all_files) > max_frames:
        all_files = all_files[:max_frames]
    
    frames = []
    for filename in all_files:
        img_path = os.path.join(folder_path, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            if resize_short is not None:
                w, h = img.size
                scale = resize_short / min(w, h)
                if scale < 1.0:
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            frames.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
    return frames

def main():
    args = parse_args()
    model, tokenizer = load_model(args.model_path)
    
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

    for folder_name in subfolders:
        folder_path = os.path.join(args.image_folder, folder_name)
        print(f"Processing folder: {folder_name}...")
        
        # 1. Retrieve the specific query for this folder
        # If folder ID isn't in our map, default to generic instruction
        current_query = QUERY_DATA.get(str(folder_name), "")
        
        # 2. Construct Dynamic Prompt
        if current_query:
            print(f"  > Focusing on: '{current_query}'")
            prompt_text = (
                f"<image> Objectively analyze the visual scene, focusing specifically on the elements mentioned in: '{current_query}'. "
                "1. Identify these main objects and describe their precise location (e.g., foreground, background, left, right, center). "
                "2. Specify distinct visual attributes for each, such as colors or unique actions. "
                "3. If multiple similar objects are present, explicitly explain the visual feature or position that distinguishes one from the other."
            )
        else:
            # Fallback if no query exists for this ID
            print(f"  > No specific query found for {folder_name}, using generic prompt.")
            prompt_text = (
                "<image> Objectively analyze the visual scene. "
                "1. Identify the main objects and describe their precise location. "
                "2. Specify distinct visual attributes for each object. "
                "3. Distinguish between similar objects using position or color."
                "4. Dont use words like group, always estimate the number of each similar objects"
            )

        frames = get_images_from_folder(folder_path, max_frames=args.max_frames, resize_short=args.resize_short_edge)
        
        if not frames:
            print(f"  No valid images found, skipping.")
            continue
            
        try:
            with torch.inference_mode():
                result = model.predict_forward(
                    video=frames,
                    text=prompt_text,
                    tokenizer=tokenizer
                )
            
            description = result['prediction']
            print(f"  > Generated: {description}...")
            
            final_results[folder_name] = {
                "FOCUS_QUERY": current_query,
                "IMAGE_DESCRIPTION": description
            }
            
        except Exception as e:
            print(f"  Error processing folder {folder_name}: {e}")
            if "out of memory" in str(e).lower():
                 print("  ! GPU OOM. Restart with --max_frames 8 --resize_short_edge 448")
                 torch.cuda.empty_cache()

        del frames
        if 'result' in locals(): del result
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":
    main()