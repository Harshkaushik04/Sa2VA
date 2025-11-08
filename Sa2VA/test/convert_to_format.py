import pandas as pd
import json
import os
import glob
from tqdm import tqdm
from pathlib import Path

# --- SCRIPT SETTINGS ---
# This script is now "general". It finds all paths relative to itself.
# It assumes this script is in: .../Sa2VA/test/
# And that the project structure is:
# .../Sa2VA/
#    +-- data/
#        +-- Ref-coco/         <-- INPUT
#        |   +-- *.parquet
#        +-- updated_Ref-coco/ <-- OUTPUT
#            +-- images/
#            +-- annotations.json
#    +-- test/
#        +-- convert_to_format.py (this file)

# 1. Get the script's own directory (e.g., .../Sa2VA/test)
SCRIPT_DIR = Path(__file__).parent
# 2. Get the project root (e.g., .../Sa2VA)
PROJECT_ROOT = SCRIPT_DIR.parent

# 3. Define Input and Output paths relative to the project root
INPUT_DIR = PROJECT_ROOT / "data" / "Ref-coco"
OUTPUT_DIR = PROJECT_ROOT / "data" / "updated_Ref-coco"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "images"
JSON_OUTPUT_FILE = OUTPUT_DIR / "annotations.json" # Plural to match finetune config

# --- DATASET-SPECIFIC SETTINGS ---
# These column names are specific to the RefCOCO parquet files
FILENAME_COL = 'file_name'
SEGMENTATION_COL = 'segmentation'
TEXT_COL = 'answer'
IMAGE_BYTES_COL = 'image'
# -------------------------------------

# 1. Create the output image directory
os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)
print(f"Created/found image directory: {IMAGE_OUTPUT_DIR}/")

# 2. Automatically find all .parquet files in the input directory
parquet_search_path = str(INPUT_DIR / "*.parquet")
PARQUET_FILES = glob.glob(parquet_search_path)

if not PARQUET_FILES:
    print(f"Error: No .parquet files found in {INPUT_DIR}")
    print("Please make sure the Ref-coco data was downloaded correctly.")
    exit(1)

print(f"Found {len(PARQUET_FILES)} Parquet files to process:")
for f in PARQUET_FILES:
    print(f"  - {os.path.basename(f)}")

print(f"Loading {len(PARQUET_FILES)} Parquet files. This might take a moment...")
# Pandas will read all files and combine them into one DataFrame
df = pd.read_parquet(PARQUET_FILES)
print(f"Total annotations found: {len(df)}")

print(f"Grouping all {len(df)} annotations by image...")
grouped = df.groupby(FILENAME_COL)

final_annotations_list = []
saved_images = set()  # A set to track which images we've already saved

# Use tqdm for a progress bar
for image_filename, rows in tqdm(grouped, desc="Converting data"):
    
    # --- Part 1: Save the Image (if we haven't already) ---
    if image_filename not in saved_images:
        try:
            # Get the first row to extract the image bytes
            first_row = rows.iloc[0]
            
            # Access the bytes: assumes format {'bytes': b'...'}
            image_bytes = first_row[IMAGE_BYTES_COL]['bytes']
            
            output_image_path = os.path.join(IMAGE_OUTPUT_DIR, image_filename)
            
            with open(output_image_path, 'wb') as f:
                f.write(image_bytes)
            
            saved_images.add(image_filename)
            
        except Exception as e:
            print(f"\n[Warning] Failed to save image {image_filename}. Error: {e}")
            continue # Skip this whole image if we can't save it

    # --- Part 2: Process Annotations ---
    image_masks = []
    image_texts = []

    for _, row in rows.iterrows():
        try:
            # 1. Get the text phrase (e.g., "guy petting elephant")
            text_list = row[TEXT_COL]
            if not text_list:
                continue
            text = text_list[0] 
            
            # 2. Get the segmentation data (which is a numpy array)
            polygon_data = row[SEGMENTATION_COL]

            # 3. Handle None/empty arrays
            if polygon_data is None:
                continue
                
            # 4. Convert numpy array to a plain Python list of floats
            #    This solves the "not JSON serializable" crash.
            polygon_list = [float(p) for p in polygon_data]
            
            # 5. Now we can safely check the length
            if len(polygon_list) >= 6: # A valid polygon needs at least 3 points (x,y)
                image_masks.append([polygon_list]) # Append in the [[...]] format
                image_texts.append(text)
            
        except Exception as e:
            # This will catch errors in a single row
            print(f"\n[Warning] Skipping bad annotation in {image_filename}. Error: {e}")
            continue
            
    # Only add to final list if we successfully found masks
    if image_texts:
        final_annotations_list.append({
            "image": image_filename,
            "mask": image_masks,
            "text": image_texts
        })

print("\n--- Conversion Summary ---")
print(f"Successfully processed and saved {len(final_annotations_list)} unique images.")
print(f"Total annotations created: {sum(len(item['text']) for item in final_annotations_list)}")

# Save the final JSON file
print(f"Saving annotations to {JSON_OUTPUT_FILE}...")
with open(JSON_OUTPUT_FILE, 'w') as f:
    json.dump(final_annotations_list, f)

print("\nAll done! Your 'annotations.json' and 'images/' folder are ready.")