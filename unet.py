import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# ------------------------
# Paths
# ------------------------
image_folder = r"C:\Users\aayus\OneDrive\Desktop\s_dataset\preflood_dataset"  # Change path if needed
output_folder = "pre_processed_images_unet"
output_csv = "sar_labels_unet.csv"
unet_model_path = "unet_model.h5"  # Path to your trained U-Net model

# ------------------------
# Load U-Net model
# ------------------------
model = load_model(unet_model_path)

# ------------------------
# Create output directory
# ------------------------
os.makedirs(output_folder, exist_ok=True)

# ------------------------
# Image processing
# ------------------------
image_labels = []

for image_file in os.listdir(image_folder):
    if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue  # Skip non-image files

    image_path = os.path.join(image_folder, image_file)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Skipping {image_file}: Unable to load.")
        continue

    # Resize image for U-Net input (Assuming model expects 128x128, change if different)
    original_size = img.shape[:2]
    input_size = (128, 128)

    resized_img = cv2.resize(img, input_size)
    resized_img = resized_img.astype('float32') / 255.0  # Normalize to [0, 1]
    resized_img = np.expand_dims(resized_img, axis=-1)   # Add channel dimension
    resized_img = np.expand_dims(resized_img, axis=0)    # Add batch dimension

    # Predict segmentation mask
    pred_mask = model.predict(resized_img)[0, :, :, 0]  # Remove batch and channel dims
    pred_mask = (pred_mask > 0.5).astype(np.uint8)       # Threshold to binary mask

    # Resize mask back to original image size
    pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)

    # Calculate water percentage
    water_percentage = (np.sum(pred_mask == 1) / pred_mask.size) * 100

    # Set label based on water percentage
    label = "Flooded" if water_percentage > 30 else "Non-Flooded"

    # Create a color image for visualization
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Apply color coding
    color_img[pred_mask == 1] = [255, 0, 0]   # Water → Blue
    color_img[pred_mask == 0] = [0, 255, 0]   # Land → Green

    # Overlay text
    cv2.putText(color_img, f"Inundation: {water_percentage:.2f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Save processed image
    processed_path = os.path.join(output_folder, image_file)
    cv2.imwrite(processed_path, color_img)

    # Save image data
    image_labels.append([image_file, label, round(water_percentage, 2)])

# ------------------------
# Save labels to CSV
# ------------------------
labels_df = pd.DataFrame(image_labels, columns=["Image", "Label", "Inundation (%)"])
labels_df.to_csv(output_csv, index=False)

print(f"✅ Processed images saved in: {output_folder}")
print(f"✅ Labels and inundation data saved in: {output_csv}")