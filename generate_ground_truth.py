#!/usr/bin/env python3
"""
Generate binary flood-extent masks (ground truth) by change detection 
between pre- and post-flood SAR image pairs.

This script:
1. Reads paired images from preflood_dataset/ and postflood_dataset/
2. Applies Otsu thresholding to identify water in each image
3. Computes flood mask as: post_water AND (NOT pre_water)
4. Applies morphological operations to clean noise
5. Saves binary masks to ground_truth_masks/

Usage:
    python generate_ground_truth.py
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from skimage.morphology import remove_small_objects, disk, opening as morph_open
from skimage.filters import threshold_otsu

# Configuration
PREFLOOD_DIR = Path("preflood_dataset")
POSTFLOOD_DIR = Path("postflood_dataset")
OUT_DIR = Path("ground_truth_masks")
VISUALIZATION_DIR = Path("ground_truth_visualizations")

# Create output directories
OUT_DIR.mkdir(exist_ok=True)
VISUALIZATION_DIR.mkdir(exist_ok=True)

def water_mask_otsu(img: np.ndarray) -> np.ndarray:
    """
    Extract water mask from SAR image using Otsu thresholding.
    
    In SAR images, water appears dark due to specular reflection.
    We use Otsu to find the optimal threshold and consider pixels
    below this threshold as water.
    """
    # Apply Otsu thresholding (water = dark pixels)
    thresh_val = threshold_otsu(img)
    water_mask = img < thresh_val
    
    # Convert to uint8 for morphological operations
    water_mask = water_mask.astype(np.uint8)
    
    # Clean speckle noise with morphological operations
    # Remove small objects (< 64 pixels)
    water_mask = remove_small_objects(water_mask.astype(bool), min_size=64)
    
    # Apply morphological opening to smooth boundaries
    water_mask = morph_open(water_mask, disk(2))
    
    return water_mask.astype(np.uint8)

def create_visualization(pre_img, post_img, pre_water, post_water, flood_mask, filename):
    """Create a visualization showing the change detection process."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pre_img, cmap='gray')
    axes[0].set_title('Pre-flood SAR')
    axes[0].axis('off')
    axes[1].imshow(post_img, cmap='gray')
    axes[1].set_title('Post-flood SAR')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(VISUALIZATION_DIR / f"{filename.stem}_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()

def calculate_statistics(pre_water, post_water, flood_mask):
    """Calculate flood statistics."""
    total_pixels = pre_water.size
    pre_water_pixels = np.sum(pre_water)
    post_water_pixels = np.sum(post_water)
    flood_pixels = np.sum(flood_mask)
    
    pre_water_pct = (pre_water_pixels / total_pixels) * 100
    post_water_pct = (post_water_pixels / total_pixels) * 100
    flood_pct = (flood_pixels / total_pixels) * 100
    
    return {
        'pre_water_percentage': pre_water_pct,
        'post_water_percentage': post_water_pct,
        'flood_percentage': flood_pct,
        'flood_pixels': flood_pixels
    }

def main():
    """Main function to process all image pairs and generate ground truth."""
    print("🌊 Generating Ground Truth Flood Masks")
    print("=" * 50)
    
    # Get list of files
    preflood_files = sorted([f for f in os.listdir(PREFLOOD_DIR) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    statistics = []
    processed_count = 0
    skipped_count = 0
    
    for filename in tqdm(preflood_files, desc="Processing image pairs"):
        pre_path = PREFLOOD_DIR / filename
        post_path = POSTFLOOD_DIR / filename
        
        # Check if both images exist
        if not post_path.exists():
            print(f"⚠️  Skipping {filename}: no matching post-flood image")
            skipped_count += 1
            continue
            
        # Load images
        pre_img = cv2.imread(str(pre_path), cv2.IMREAD_GRAYSCALE)
        post_img = cv2.imread(str(post_path), cv2.IMREAD_GRAYSCALE)
        
        if pre_img is None or post_img is None:
            print(f"⚠️  Skipping {filename}: cannot read images")
            skipped_count += 1
            continue
            
        # Ensure images have the same dimensions
        if pre_img.shape != post_img.shape:
            print(f"⚠️  Skipping {filename}: image dimensions don't match")
            skipped_count += 1
            continue
        
        # Extract water masks
        pre_water = water_mask_otsu(pre_img)
        post_water = water_mask_otsu(post_img)
        
        # Compute flood mask: new water = post_water AND (NOT pre_water)
        flood_mask = post_water & (~pre_water.astype(bool))
        flood_mask = flood_mask.astype(np.uint8)
        
        # Save binary mask (0 = no flood, 255 = flood)
        flood_mask_save = flood_mask * 255
        output_path = OUT_DIR / filename
        cv2.imwrite(str(output_path), flood_mask_save)
        
        # Calculate statistics
        stats = calculate_statistics(pre_water, post_water, flood_mask)
        stats['filename'] = filename
        statistics.append(stats)
        
        # Create visualization for first few images
        if processed_count < 5:
            create_visualization(pre_img, post_img, pre_water, post_water, 
                               flood_mask, Path(filename))
        
        processed_count += 1
    
    # Save statistics
    if statistics:
        stats_df = pd.DataFrame(statistics)
        stats_df.to_csv('ground_truth_statistics.csv', index=False)
        
        print(f"\n📊 Processing Summary:")
        print(f"   ✅ Successfully processed: {processed_count} image pairs")
        print(f"   ⚠️  Skipped: {skipped_count} files")
        print(f"   📁 Ground truth masks saved to: {OUT_DIR.resolve()}")
        print(f"   📈 Statistics saved to: ground_truth_statistics.csv")
        
        # Print summary statistics
        avg_flood_pct = stats_df['flood_percentage'].mean()
        max_flood_pct = stats_df['flood_percentage'].max()
        min_flood_pct = stats_df['flood_percentage'].min()
        
        print(f"\n🔍 Flood Detection Summary:")
        print(f"   Average flood coverage: {avg_flood_pct:.2f}%")
        print(f"   Maximum flood coverage: {max_flood_pct:.2f}%")
        print(f"   Minimum flood coverage: {min_flood_pct:.2f}%")
        
        # Show most flooded images
        top_flooded = stats_df.nlargest(3, 'flood_percentage')
        print(f"\n🌊 Most flooded areas:")
        for _, row in top_flooded.iterrows():
            print(f"   {row['filename']}: {row['flood_percentage']:.2f}%")
            
        print(f"\n🖼️  Sample visualizations saved to: {VISUALIZATION_DIR.resolve()}")
    else:
        print("❌ No images were successfully processed!")

if __name__ == "__main__":
    main() 