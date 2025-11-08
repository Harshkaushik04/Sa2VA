#!/bin/bash
set -e # Exit immediately if any command fails

echo "--- Sa2VA Project Setup Script ---"
echo "This script will create the venv, pretrained, and data folders."
echo "This requires 'python3.11' to be available on your system."
echo "Press Enter to begin..."
read

# --- 1. VENV SETUP ---
echo "[1/3] Setting up Python virtual environment..."

echo "  > Installing uv (Python package manager)..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Source the cargo/local env to make `uv` available immediately
# This handles the case where it's a fresh install
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
elif [ -f "$HOME/.local/env" ]; then
    source "$HOME/.local/env"
fi

echo "  > Creating virtual environment at ./venv using python3.11..."
# The pyproject.toml specifies python >=3.11,<3.12
# We attempt to use 'python3.11' explicitly.
python3.11 -m venv venv
source ./venv/bin/activate

echo "  > Installing dependencies into ./venv (this will take a few minutes)..."
# We run `uv sync` from inside the Sa2VA directory so it finds the
# pyproject.toml and uv.lock files. We call the `uv` executable
# that lives inside our new venv.

# Change to the Sa2VA directory
cd Sa2VA

# Call the `uv` from our venv to install packages *into* that venv
# This command looks for `pyproject.toml` in the current dir (Sa2VA)
# and installs packages into `../venv`
uv sync --extra=legacy --active

# Go back to the parent directory
cd ..
echo "--- [1/3] VENV setup complete! ---"
echo ""

# --- 2. PRETRAINED MODELS ---
echo "[2/3] Downloading pretrained models..."

echo "  > Creating directory: ./Sa2VA/pretrained/sam2"
mkdir -p Sa2VA/pretrained/sam2

echo "  > Downloading SAM2-Hiera-Large (sam2_hiera_large.pt)..."
# We download the file directly into the correct subfolder with the correct name
wget -O Sa2VA/pretrained/sam2/sam2_hiera_large.pt https://huggingface.co/facebook/sam2-hiera-large/resolve/main/sam2_hiera_large.pt

echo "  > Converting InternVL3-2B model (using venv)..."
# Run the conversion script using the venv's python
venv/bin/python Sa2VA/tools/convert_to_pth.py OpenGVLab/InternVL3-2B --save-path Sa2VA/pretrained/Sa2VA-InternVL3-2B.pth --arch-type internvl

echo "--- [2/3] PRETRAINED models setup complete! ---"
echo ""

# --- 3. DATASET ---
echo "[3/3] Downloading and preparing dataset..."

# Define the base URL for wget (using the correct /resolve/main/ path)
BASE_URL="https://huggingface.co/datasets/lmms-lab/RefCOCO/resolve/main"

# Define the destination folder
DEST_FOLDER="Sa2VA/data/Ref-coco"

echo "  > Creating directory: $DEST_FOLDER"
mkdir -p "$DEST_FOLDER"

# Define the array of file names (as seen in your error log)
declare -a PARQUET_FILES=(
    "test-00000-of-00002.parquet"
    "test-00001-of-00002.parquet"
    "testA-00000-of-00001.parquet"
    "testB-00000-of-00001.parquet"
    "val-00000-of-00004.parquet"
    "val-00001-of-00004.parquet"
    "val-00002-of-00004.parquet"
    "val-00003-of-00004.parquet"
)

echo "  > Downloading RefCOCO parquet files (8 files)..."

# Loop through the array and download each file
for item in "${PARQUET_FILES[@]}"; do
    # The file is in the 'data/' subfolder on the repo
    REMOTE_PATH="data/$item"
    FULL_URL="$BASE_URL/$REMOTE_PATH"
    
    # Construct the full local destination path
    # The files will be saved directly inside DEST_FOLDER
    LOCAL_PATH="$DEST_FOLDER/$item"
    
    echo "    > Downloading $item..."
    # Use -c to allow resuming, -O for the output file
    wget -c -O "$LOCAL_PATH" "$FULL_URL"
done

echo "  > All parquet files downloaded."

echo "  > Running convert_to_format.py (using venv)..."
# Change into Sa2VA so the scripts run from its expected location
(
    cd Sa2VA
    ../venv/bin/python3 test/convert_to_format.py
)
rm -rf data/Ref-coco

echo "--- [3/3] DATA setup complete! ---"
echo ""
echo "✅ All done! You can now activate the environment with:"
echo "source venv/bin/activate"
