import argparse
import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig

import torch

import cv2
try:
    from mmengine.visualization import Visualizer
except (ImportError, RuntimeError):
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

def split_model(model_path):
    import math
    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = num_gpus // world_size

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    print(f"Model {model_path} has {num_layers} layers.")

    # Since the first GPU will be used for ViT, treat it as 0.5 GPU.
    num_layers_per_gpu = math.ceil(num_layers / (num_gpus - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    print(f"num_layers_per_gpu: {num_layers_per_gpu}")
    
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
            layer_cnt += 1
    
    device_map['vision_model'] = rank
    device_map['mlp1'] = rank
    device_map['language_model.model.tok_embeddings'] = rank
    device_map['language_model.model.embed_tokens'] = rank
    device_map['language_model.output'] = rank
    device_map['language_model.model.norm'] = rank
    device_map['language_model.lm_head'] = rank
    device_map[f'language_model.model.layers.{num_layers - 1}'] = rank
    device_map['grounding_encoder'] = rank
    device_map['text_hidden_fcs'] = rank

    return device_map

def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('image_folder', help='Path to image file')
    parser.add_argument('--model_path', default="ByteDance/Sa2VA-B")
    parser.add_argument('--work-dir', default=None, help='The dir to save results.')
    parser.add_argument('--text', type=str, default="<image>Please describe the video content.")
    parser.add_argument('--select', type=int, default=-1)
    args = parser.parse_args()
    return args


def visualize(pred_mask, image_path, work_dir):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='g', alphas=0.4)
    visual_result = visualizer.get_image()

    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)

if __name__ == "__main__":
    cfg = parse_args()
    model_path = cfg.model_path
    
    # --- HERE ARE THE CHANGES ---
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # Changed from "auto" for performance
        device_map="auto",         # Uncommented to load model on GPU
        trust_remote_code=True
    )
    # --- END OF CHANGES ---
    
    """
    # For distributed inference, uncomment the following lines to get device_map
    device_map=split_model(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True
    )
    """

    if 'qwen' in model_path.lower():
        print("Using AutoProcessor for Qwen-VL model.")
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        tokenizer = None
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )


    image_files = []
    image_paths = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}
    for filename in sorted(list(os.listdir(cfg.image_folder))):
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_files.append(filename)
            image_paths.append(os.path.join(cfg.image_folder, filename))

    vid_frames = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        vid_frames.append(img)


    if cfg.select > 0:
        img_frame = vid_frames[cfg.select - 1]

        print(f"Selected frame {cfg.select}")
        print(f"The input is:\n{cfg.text}")
        result = model.predict_forward(
            image=img_frame,
            text=cfg.text,
            tokenizer=tokenizer,
            # processor=processor, # This line remains commented out
        ) # type: ignore
    else:
        print(f"The input is:\n{cfg.text}")
        result = model.predict_forward(
            video=vid_frames,
            text=cfg.text,
            tokenizer=tokenizer,
            # processor=processor, # This line remains commented out
        ) # type: ignore

    prediction = result['prediction']
    print(f"The output is:\n{prediction}")

    if '[SEG]' in prediction and Visualizer is not None:
        
        # result['prediction_masks'] is a LIST of mask arrays. 
        # Each item in the list corresponds to one [SEG] token.
        # Each mask array has shape (num_frames, H, W).
        all_mask_sets = result['prediction_masks']
        
        print(f"Found {len(all_mask_sets)} segmentation mask set(s).")

        for frame_idx in range(len(vid_frames)):
            # For single image mode, only process the selected frame
            if cfg.select > 0 and frame_idx != (cfg.select - 1):
                continue

            masks_for_this_frame = []
            for mask_set in all_mask_sets:
                # Get the mask for the current frame from this mask set
                if frame_idx < len(mask_set):
                    masks_for_this_frame.append(mask_set[frame_idx])
            
            if not masks_for_this_frame:
                continue # No masks for this frame

            # Stack all masks for this frame into a single numpy array (N, H, W)
            combined_masks_for_frame = np.stack(masks_for_this_frame, axis=0)

            # --- In-lining the visualize function to handle multiple masks ---
            visualizer = Visualizer()
            img = cv2.imread(image_paths[frame_idx])
            visualizer.set_image(img)
            
            # Draw all masks with random colors
            # This is the key change: draw_binary_masks can handle a stack of masks
            visualizer.draw_binary_masks(combined_masks_for_frame, colors='random', alphas=0.4)
            visual_result = visualizer.get_image()

            # --- Original saving logic ---
            output_path_dir = cfg.work_dir if cfg.work_dir else './temp_visualize_results'
            os.makedirs(output_path_dir, exist_ok=True)
            output_path = os.path.join(output_path_dir, os.path.basename(image_paths[frame_idx]))
            cv2.imwrite(output_path, visual_result)
            print(f"Saved visualization for frame {frame_idx} to {output_path}")
            
    else:
        pass