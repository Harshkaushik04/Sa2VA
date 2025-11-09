import pandas as pd
import json
import os
import numpy as np
from pycocotools import mask as mask_utils
from skimage import measure
from tqdm import tqdm
import glob

# --- TO-DO: EDIT THESE ---

# 1. Define the input directory (relative to where you run this script)
INPUT_DIR = 'data/Ref-coco/data' 

# 2. Define your output directory (relative to where you run this script)
OUTPUT_DIR = 'data/refcoco_converted'

# 3. Define the Parquet column names (these are likely correct)
FILENAME_COL = 'file_name'
IMAGE_BYTES_COL = 'image'
SEGMENTATION_COL = 'segmentation' # This is the RLE object
TEXT_COL = 'answer' # We'll take the first answer as the text
# -------------------------------------

# --- SCRIPT SETTINGS ---
IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'images')
JSON_OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'annotations.json')

# --- Helper Function: RLE to Polygons ---
def rle_to_polygons(rle_obj):
    if not rle_obj or not rle_obj.get('counts'):
        return []
    try:
        binary_mask = mask_utils.decode(rle_obj)
    except Exception as e:
        print(f"Error decoding RLE: {e}")
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
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
    print(f"Output directory created at: {OUTPUT_DIR}")

    # --- Automatically find all .parquet files ---
    print(f"Searching for .parquet files in: {INPUT_DIR}")
    search_path = os.path.join(INPUT_DIR, '*.parquet')
    PARQUET_FILES = sorted(glob.glob(search_path)) # Sort to ensure order
    
    if not PARQUET_FILES:
        print(f"Error: No .parquet files found at '{search_path}'.")
        print("Please check the INPUT_DIR path in the script.")
        return
    
    print(f"Found {len(PARQUET_FILES)} Parquet files to process one by one.")
    
    total_annotated_images = 0
    saved_images = set()

    # Open the output file *first* and write line by line
    with open(JSON_OUTPUT_FILE, 'w') as f:
        f.write('[\n') # Start the JSON list
        is_first_item_in_all_files = True # Flag to handle commas

        # --- NEW: Loop through each file ---
        for parquet_file in PARQUET_FILES:
            print(f"\nProcessing file: {parquet_file}")
            
            try:
                df = pd.read_parquet(parquet_file)
            except Exception as e:
                print(f"  [Warning] Failed to read {parquet_file}. Error: {e}. Skipping.")
                continue

            print("  Grouping annotations by image...")
            grouped = df.groupby(FILENAME_COL)

            for image_filename, rows in tqdm(grouped, desc="  Converting data"):
                
                # --- Part 1: Save the Image ---
                if image_filename not in saved_images:
                    try:
                        first_row = rows.iloc[0]
                        image_bytes = first_row[IMAGE_BYTES_COL]['bytes']
                        output_image_path = os.path.join(IMAGE_OUTPUT_DIR, image_filename)
                        with open(output_image_path, 'wb') as img_f:
                            img_f.write(image_bytes)
                        saved_images.add(image_filename)
                    except Exception as e:
                        print(f"\n  [Warning] Failed to save image {image_filename}. Error: {e}")
                        continue

                # --- Part 2: Process Annotations ---
                frame_masks = []
                frame_texts = []
                
                for _, row in rows.iterrows():
                    try:
                        text_list = row[TEXT_COL]
                        if not text_list:
                            continue
                        text = text_list[0]
                        
                        rle_obj = row[SEGMENTATION_COL]
                        polygons = rle_to_polygons(rle_obj)
                        
                        if polygons:
                            frame_masks.append(polygons)
                            frame_texts.append(text)
                    except Exception as e:
                        print(f"\n  [Warning] Skipping bad annotation in {image_filename}. Error: {e}")
                        continue

                # --- Part 3: Stream to JSON File ---
                if frame_texts:
                    if not is_first_item_in_all_files:
                        f.write(',\n') # Add a comma before this new item
                    
                    frame_data = {
                        "image": image_filename,
                        "mask": frame_masks,
                        "text": frame_texts
                    }
                    json.dump(frame_data, f)
                    is_first_item_in_all_files = False
                    total_annotated_images += 1
            
            # --- End of file processing ---
            print(f"  Finished processing {parquet_file}.")

        f.write('\n]\n') # Close the JSON list

    print("\n--- Conversion Summary ---")
    print(f"Successfully processed and saved {len(saved_images)} images.")
    print(f"Created {total_annotated_images} image entries in {JSON_OUTPUT_FILE}.")
    print("\nAll done!")

if __name__ == "__main__":
    main()