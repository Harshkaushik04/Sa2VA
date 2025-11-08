import os
from huggingface_hub import HfApi, HfFileSystem

# --- Settings ---
DATASET_REPO = "lmms-lab/RefCOCO"
# This is the line you wanted to change:
LOCAL_DIR = "./data/Ref-coco"
FILE_PATTERN = "*.parquet"
REPO_SUBFOLDER = "data" 
# ----------------

print(f"Starting download for '{DATASET_REPO}'...")
print(f"Looking inside subfolder: '{REPO_SUBFOLDER}'")
print(f"Will save files to: {LOCAL_DIR}")

# Create the API client
api = HfApi()

try:
    fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
    
    # Corrected glob pattern to search inside the REPO_SUBFOLDER
    glob_path = f"datasets/{DATASET_REPO}/{REPO_SUBFOLDER}/{FILE_PATTERN}"
    all_file_paths = fs.glob(glob_DATH)
    
    # Get the relative paths inside the repo, e.g., "data/train-00000-of-....parquet"
    # We strip the "datasets/lmms-lab/RefCOCO/" part
    repo_path_prefix = f"datasets/{DATASET_REPO}/"
    files_to_download = [path.replace(repo_path_prefix, "") for path in all_file_paths]
    
    if not files_to_download:
        print(f"Error: No files matching '{FILE_PATTERN}' found in '{REPO_SUBFOLDER}'.")
        exit()

    print(f"Found {len(files_to_download)} files to download:")
    for f in files_to_download:
        print(f"  - {f}")

    # Download each file
    for repo_file_path in files_to_download:
        print(f"\nDownloading {repo_file_path}...")
        api.hf_hub_download(
            repo_id=DATASET_REPO,
            filename=repo_file_path,  # Pass the full relative path, e.g., "data/file.parquet"
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded {repo_file_path}")

    print("\nAll files downloaded successfully!")
    print(f"Check your folder: ./{LOCAL_DIR}/{REPO_SUBFOLDER}")

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please check your internet connection and the repository name/paths.")