#!/bin/bash

# Download svla_so101_pickplace dataset from Hugging Face
# This script downloads the dataset and saves it to the local svla_so101_pickplace folder

echo "🚀 Starting download of svla_so101_pickplace dataset from Hugging Face..."

# Set the dataset name and local directory
DATASET_NAME="lerobot/svla_so101_pickplace"
LOCAL_DIR="svla_so101_pickplace2"

# Create local directory if it doesn't exist
echo "📁 Creating local directory: $LOCAL_DIR"
mkdir -p "$LOCAL_DIR"

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "📦 Installing huggingface_hub..."
    pip install huggingface_hub
fi

# Download the dataset using huggingface_hub
echo "⬇️  Downloading dataset from: $DATASET_NAME"
python -c "
from huggingface_hub import snapshot_download
import os

# Download the dataset
local_path = snapshot_download(
    repo_id='$DATASET_NAME',
    repo_type='dataset',
    local_dir='$LOCAL_DIR',
    local_dir_use_symlinks=False
)

print(f'✅ Dataset downloaded successfully to: {local_path}')
"

# Verify the download
if [ -d "$LOCAL_DIR" ]; then
    echo "✅ Download completed successfully!"
    echo "📊 Dataset contents:"
    ls -la "$LOCAL_DIR"
    
    # Check for key files
    if [ -f "$LOCAL_DIR/meta/info.json" ]; then
        echo "✅ Found meta/info.json"
    fi
    
    if [ -d "$LOCAL_DIR/data" ]; then
        echo "✅ Found data directory"
        echo "📁 Data files:"
        ls -la "$LOCAL_DIR/data"
    fi
    
    if [ -d "$LOCAL_DIR/videos" ]; then
        echo "✅ Found videos directory"
        echo "📹 Video files:"
        ls -la "$LOCAL_DIR/videos"
    fi
    
    echo ""
    echo "🎉 Dataset download complete!"
    echo "📍 Location: $(pwd)/$LOCAL_DIR"
    echo "📖 You can now use this dataset for training with LeRobot"
    
else
    echo "❌ Download failed! Please check the error messages above."
    exit 1
fi 