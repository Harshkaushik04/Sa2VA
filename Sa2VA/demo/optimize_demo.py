import argparse
import os
import torch
import cv2
import gc
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

# Try importing mmengine for visualization
try:
    from mmengine.visualization import Visualizer
except (ImportError, RuntimeError):
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")

def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation (Optimized)')
    parser.add_argument('image_folder', help='Path to folder containing video frames')
    parser.add_argument('--model_path', default="ByteDance/Sa2VA-B")
    parser.add_argument('--work-dir', default=None, help='The dir to save results.')
    parser.add_argument('--text', type=str, default="<image>Please describe the video content.")
    
    # OPTIONAL OPTIMIZATIONS (Defaults = None = Full Quality/All Frames)
    parser.add_argument('--max_frames', type=int, default=None, help='Optional: Limit frame count (e.g. 8). If not set, processes ALL frames.')
    parser.add_argument('--resize_short_edge', type=int, default=None, help='Optional: Resize short edge (e.g. 448). If not set, uses original resolution.')
    
    parser.add_argument('--select', type=int, default=-1)
    args = parser.parse_args()
    return args

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
    """
    Loads frames. Only downsamples or resizes if arguments are explicitly provided.
    """
    total_frames = len(image_paths)
    
    # 1. Temporal Sampling (Only if requested)
    if max_frames is not None and total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        sampled_paths = [image_paths[i] for i in indices]
        print(f"Optimization: Sampling {max_frames} frames out of {total_frames}.")
    else:
        # Default: Load ALL frames
        sampled_paths = image_paths
        
    loaded_frames = []
    for p in sampled_paths:
        try:
            img = Image.open(p).convert('RGB')
            
            # 2. Spatial Resizing (Only if requested)
            if resize_short is not None:
                w, h = img.size
                scale = resize_short / min(w, h)
                if scale < 1.0: # Only downscale
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            loaded_frames.append(img)
        except Exception as e:
            print(f"Warning: Failed to load {p}: {e}")
            
    return loaded_frames, sampled_paths

def visualize(pred_mask, image_path, work_dir, loaded_frame=None):
    if Visualizer is None: return

    visualizer = Visualizer()
    
    # Use the loaded frame if available (matches mask size), otherwise reload original
    if loaded_frame is not None:
        img = np.array(loaded_frame)
        # Convert RGB (PIL) to BGR (OpenCV)
        img = img[:, :, ::-1].copy() 
    else:
        img = cv2.imread(image_path)

    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()

    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    cfg = parse_args()
    model_path = cfg.model_path
    
    print(f"Loading model: {model_path} (4-bit quantization)...")
    
    # 1. 4-bit Configuration (Crucial for 8GB VRAM)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 2. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # 3. Load Tokenizer/Processor
    if 'qwen' in model_path.lower():
        print("Using AutoProcessor for Qwen-VL model.")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = None
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 4. Prepare Images
    all_image_paths = get_image_files(cfg.image_folder)
    
    if not all_image_paths:
        print("No images found in folder.")
        exit()

    # Load frames based on args (defaults to ALL frames, FULL resolution)
    vid_frames, processed_paths = load_and_process_frames(
        all_image_paths, 
        max_frames=cfg.max_frames, 
        resize_short=cfg.resize_short_edge
    )

    # Garbage collection before inference
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Run Inference
    print(f"Running inference on {len(vid_frames)} frames...")
    
    # use inference_mode to save VRAM (gradients are not needed)
    with torch.inference_mode():
        if cfg.select > 0:
            # Single Frame Selection Mode
            # Matches user's original logic: select based on folder index
            original_idx = cfg.select - 1
            if 0 <= original_idx < len(all_image_paths):
                target_path = all_image_paths[original_idx]
                
                # Load this specific frame freshly (to ensure resizing consistency if used)
                single_frame_list, single_path_list = load_and_process_frames(
                    [target_path], 
                    resize_short=cfg.resize_short_edge
                )
                img_frame = single_frame_list[0]
                
                print(f"Selected frame {cfg.select}: {os.path.basename(target_path)}")
                print(f"Input: {cfg.text}")
                
                result = model.predict_forward(
                    image=img_frame,
                    text=cfg.text,
                    tokenizer=tokenizer,
                )
                # Prepare list for visualization logic
                processed_paths = single_path_list
                vid_frames = single_frame_list
            else:
                print(f"Error: Selection {cfg.select} is out of range.")
                exit()
        else:
            # Video Mode
            print(f"Input: {cfg.text}")
            try:
                result = model.predict_forward(
                    video=vid_frames,
                    text=cfg.text,
                    tokenizer=tokenizer,
                )
            except torch.cuda.OutOfMemoryError:
                print("\n!!! CRITICAL ERROR: GPU Out of Memory !!!")
                print("You are trying to process too many frames or too high resolution.")
                print("Please re-run with optimizations, for example:")
                print(f"python script.py {cfg.image_folder} --max_frames 8 --resize_short_edge 448")
                exit()

    prediction = result['prediction']
    print(f"Output: {prediction}")

    # 6. Visualization
    if '[SEG]' in prediction and Visualizer is not None:
        _seg_idx = 0
        if 'prediction_masks' in result and len(result['prediction_masks']) > 0:
            pred_masks = result['prediction_masks'][_seg_idx]
            
            output_dir = cfg.work_dir if cfg.work_dir else './temp_visualize_results'
            os.makedirs(output_dir, exist_ok=True)
            
            # Align masks with the processed frames
            for i, frame_path in enumerate(processed_paths):
                if i < len(pred_masks):
                    mask = pred_masks[i]
                    visualize(mask, frame_path, output_dir, loaded_frame=vid_frames[i])
        else:
            print("Warning: [SEG] detected but no mask data returned.")