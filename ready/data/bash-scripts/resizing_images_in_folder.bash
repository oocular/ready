#!/bin/bash

# Script to resize all images in a directory to HEIGHTxWIDTH pixels
# Argument order: [image directory] [height] [width]
# Requires ImageMagick to be installed

# Default directory is current directory
DIR="${1:-.}"
HEIGHT="$2"
WIDTH="$3" 

# Check if directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist."
    exit 1
fi

# Create output directory for resized images
OUTPUT_DIR="$DIR/resized"
mkdir -p "$OUTPUT_DIR"

# Supported image extensions
extensions=("jpg" "jpeg" "png")

echo "Resizing images in directory: $DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Target size: HEIGHTxWIDTH"
echo "Will convert images to $2x$3"
echo "####################"

# Process each image type
for ext in "${extensions[@]}"; do
    # Process both lowercase and uppercase extensions
    for file in "$DIR"/*."$ext" "$DIR"/*."${ext^^}"; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "Processing: $filename"
            
            # Resize image maintaining aspect ratio, then crop/pad to exact dimensions
            convert "$file" -resize ${HEIGHT}x${WIDTH}^ -gravity center -extent ${HEIGHT}x${WIDTH} "$OUTPUT_DIR/$filename"
            
            if [ $? -eq 0 ]; then
                echo "Successfully resized"
            else
                echo "Failed to resize"
            fi
        fi
    done
done

echo "###################"
echo "Processing complete. Images resized."

# Ask if user wants to replace original files
read -p "Do you want to replace the original files with resized versions? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Replacing original files..."
    for file in "$OUTPUT_DIR"/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            mv "$file" "$DIR/$filename"
            echo "Replaced: $filename"
        fi
    done
    rmdir "$OUTPUT_DIR"
    echo "Original files have been replaced."
else
    echo "Original files preserved. Resized images are in: $OUTPUT_DIR"
fi
