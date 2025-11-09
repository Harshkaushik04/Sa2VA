import os
import json
import tarfile
import shutil
import numpy as np
from pycocotools import mask as mask_utils
from skimage import measure
from tqdm import tqdm

# --- Configuration ---
OUTPUT_DIR = 'data/mevis-testing'
DATASET_SUBFOLDER = 'valid_u'
IMAGE_DIR = os.path.join(OUTPUT_DIR, 'images')
JSON_PATH = os.path.join(OUTPUT_DIR, 'annotations.json')
TEMP_DIR = os.path.join(OUTPUT_DIR, 'temp_extracted')

# --- Helper Function: RLE to Polygons ---
def rle_to_polygons(rle_obj):
    if not rle_obj or not rle_obj.get('counts'):
        return []
    try:
        binary_mask = mask_utils.decode(rle_obj)
    except Exception:
        return []
    contours = measure.find_contours(binary_mask, 0.5)
    output_polygons = []
    for contour in contours:
        contour = np.flip(contour, axis=1)
        polygon = contour.ravel().tolist()
        if len(polygon) >= 8:
            output_polygons.append(polygon)
    return output_polygons

# --- Main Script ---
def main():
    print(f"Creating output directory: {OUTPUT_DIR}")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    all_masks = {}
    all_metas = {}

    # --- 1. Load and Extract Local Data ---
    print(f"\nProcessing dataset: {DATASET_SUBFOLDER}")
    dset_folder = os.path.join(OUTPUT_DIR, DATASET_SUBFOLDER)
    tar_path = os.path.join(dset_folder, 'JPEGImages.tar')
    mask_path = os.path.join(dset_folder, 'mask_dict.json')
    meta_path = os.path.join(dset_folder, 'meta_expressions.json')
    
    if not all(os.path.exists(p) for p in [tar_path, mask_path, meta_path]):
        print(f"Error: Could not find files in {dset_folder}")
        return
    print(f"Extracting {DATASET_SUBFOLDER} images...")
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=TEMP_DIR)
    except Exception as e:
        print(f"Error extracting TAR file: {e}")
        return
    print(f"Loading {DATASET_SUBFOLDER} JSONs...")
    try:
        with open(mask_path) as f:
            all_masks = json.load(f)
        with open(meta_path) as f:
            all_metas = json.load(f)['videos']
    except Exception as e:
        print(f"Error reading JSON files: {e}")
        return

    if not all_masks or not all_metas:
        print("No data was loaded. Exiting.")
        return

    # --- 2. Flatten and Rename Images ---
    print("\nFlattening image directory structure...")
    extracted_images_root = os.path.join(TEMP_DIR, 'JPEGImages')
    if not os.path.exists(extracted_images_root):
        print(f"Error: 'JPEGImages' folder not found after extraction.")
        shutil.rmtree(TEMP_DIR)
        return
    for video_id in tqdm(os.listdir(extracted_images_root), desc="Renaming images"):
        video_frame_dir = os.path.join(extracted_images_root, video_id)
        if not os.path.isdir(video_frame_dir):
            continue
        for frame_file in os.listdir(video_frame_dir):
            if frame_file.endswith('.jpg'):
                frame_name = os.path.splitext(frame_file)[0]
                new_name = f"{video_id}_{frame_name}.jpg"
                src_path = os.path.join(video_frame_dir, frame_file)
                dst_path = os.path.join(IMAGE_DIR, new_name)
                shutil.move(src_path, dst_path)
    shutil.rmtree(TEMP_DIR)
    print("Image flattening complete.")

    # --- 3. Process and Convert (Streaming to File) ---
    print(f"Processing annotations and streaming to {JSON_PATH}...")
    
    total_annotated_images = 0
    # Open the output file *first* and write line by line
    with open(JSON_PATH, 'w') as f:
        f.write('[\n') # Start the JSON list
        is_first_item = True

        for video_id, meta in tqdm(all_metas.items(), desc="Converting annotations"):
            expressions = meta['expressions']
            frames = meta['frames']
            
            for frame_index, frame_name in enumerate(frames):
                image_filename = f"{video_id}_{frame_name}.jpg"
                frame_masks = []
                frame_texts = []
                
                for exp_id, exp_data in expressions.items():
                    text = exp_data['exp']
                    anno_ids = exp_data.get('anno_id')
                    
                    if not anno_ids:
                        continue
                    
                    for anno_id in anno_ids:
                        anno_id_str = str(anno_id)
                        if anno_id_str not in all_masks:
                            continue
                        
                        obj_all_frames = all_masks[anno_id_str]
                        if frame_index >= len(obj_all_frames):
                            continue
                        
                        rle_obj = obj_all_frames[frame_index]
                        if not rle_obj or not rle_obj.get('counts'):
                            continue
                        
                        polygons = rle_to_polygons(rle_obj)
                        if polygons:
                            frame_masks.append(polygons)
                            frame_texts.append(text)

                # --- NEW STREAMING LOGIC ---
                # If we found masks for this frame, write it to the file immediately
                if frame_texts:
                    if not is_first_item:
                        f.write(',\n') # Add a comma before this new item
                    
                    frame_data = {
                        "image": image_filename,
                        "mask": frame_masks,
                        "text": frame_texts
                    }
                    json.dump(frame_data, f)
                    is_first_item = False
                    total_annotated_images += 1
        
        f.write('\n]\n') # Close the JSON list

    # --- 4. Final Report ---
    print(f"\nConverted {len(all_metas)} videos into {total_annotated_images} annotated images.")
    print("Testing data creation complete.")
    print(f"Your testing data is ready in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()