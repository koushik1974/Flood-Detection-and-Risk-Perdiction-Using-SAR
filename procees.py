import os
import cv2
import numpy as np
import pandas as pd

# Set folder paths
image_folder = r"C:\Users\aayus\OneDrive\Desktop\s_dataset\preflood_dataset"  # Update your path
output_csv = "sar_labels.csv"
output_folder = "pre_processed_images"

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# List to store image labels
image_labels = []

# Loop through each SAR image
for image_file in os.listdir(image_folder):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # Skip non-image files

    image_path = os.path.join(image_folder, image_file)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale

    if img is None:
        print(f"Skipping {image_file}: Unable to load.")
        continue

    # Apply Otsu's thresholding to detect water (dark areas)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calculate percentage of water (dark pixels)
    water_percentage = (np.sum(thresh == 0) / thresh.size) * 100  # % of dark pixels

    # Set label: Flooded if >30% dark pixels
    label = "Flooded" if water_percentage > 30 else "Non-Flooded"
    
    # Convert grayscale to BGR
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Apply color mapping: Blue for water, Green for land
    color_img[thresh == 0] = [255, 0, 0]   # Water → Blue
    color_img[thresh == 255] = [0, 255, 0]  # Land → Green

    # Overlay text with inundation percentage
    cv2.putText(color_img, f"Inundation: {water_percentage:.2f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Save processed image
    processed_path = os.path.join(output_folder, image_file)
    cv2.imwrite(processed_path, color_img)

    # Store label and inundation percentage
    image_labels.append([image_file, label, round(water_percentage, 2)])

# Save labels to CSV
labels_df = pd.DataFrame(image_labels, columns=["Image", "Label", "Inundation (%)"])
labels_df.to_csv(output_csv, index=False)

print(f"Processed images saved in {output_folder}")
print(f"Labels and inundation data saved in {output_csv}")
