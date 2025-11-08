<img width="1046" height="679" alt="image5" src="https://github.com/user-attachments/assets/687a4235-5919-4dd3-a956-268290232b1d" /># Flood-Detection-and-Risk-Perdiction-Using-SAR
This project uses pre- and post-flood SAR satellite images to analyze water spread patterns and identify flood-prone areas. By learning from past flood data, the model predicts which regions are most likely to be affected in future flood waves, aiding early warning and disaster management.


🌊 Flood Prediction Using SAR Satellite Images
🛰️ Overview

This project focuses on predicting future flood-prone areas using SAR (Synthetic Aperture Radar) satellite images. We use pre-flood and post-flood data of major flood events in India, collected from the Copernicus database.

By analyzing spatial changes in water coverage, the system learns flood behavior patterns to help identify regions that may be affected in upcoming flood waves.

🧩 Data Collection & Preprocessing

Collected pre- and post-flood SAR images from Copernicus.

Scaled images into smaller regions for better processing.

Paired and labeled corresponding pre- and post-flood tiles.

Removed noise and ensured all images are of uniform size and resolution.

🧠 Model

We train a U-Net model, well-suited for image segmentation tasks.
The model learns from paired images to detect flooded regions and predict likely future flood zones based on past patterns.

🗓️ Weekly Progress Report - Week - 1

This week, I explored information about major floods in India and identified the regions most affected. I also researched when these floods typically occur — analyzing seasonal patterns and times of the year when flooding is most frequent.

I examined the SAR satellite images in the collected dataset to understand the differences between pre-flood and post-flood conditions. Additionally, I searched for suitable models to clean and preprocess the data, which will be used for training in the upcoming week.

To support this, I developed two Python scripts:

unet.py – Labels the data based on the time and location of each flood event.

train_flood_unet.py – Creates the dataset and trains the U-Net model for flood prediction.

Next week, the focus will be on refining the dataset and training the model to evaluate its performance.
<img width="1046" height="679" alt="image5" src="https://github.com/user-attachments/assets/e4ec0aa5-f591-4664-b1b4-6b2098d69472" />
<img width="1046" height="679" alt="image5" src="https://github.com/user-attachments/assets/0ed8c505-a8e8-4a65-b317-f4107c315961" />



🗓️ Week 2 Progress — Ground Truth Generation & Model Training

In Week 2, I focused on understanding the ground truth creation process and how pixel-level changes in SAR (Synthetic Aperture Radar) images can be used to detect flood-affected regions.

🔍 Key Learnings:

Studied visualization of pixel values and how they are segmented into binary masks (0 for non-flooded, 1 for flooded regions).

Learned how SAR images differ before and after floods, and how those differences indicate water accumulation.

Implemented change detection logic to generate binary flood-extent masks, using Otsu’s thresholding and morphological operations to reduce noise.

🧠 Implementation Highlights:

Wrote a Python script generate_ground_truth.py that:

Reads pre-flood and post-flood SAR image pairs.

Applies Otsu thresholding to detect water bodies.

Generates flood masks by comparing pixel intensity changes between the two images.

Saves cleaned binary masks to the ground_truth_masks/ directory.

Produces visualizations and flood statistics (percentage of flooded pixels) for analysis.

🧩 Model Training:

Started training the flood detection model using the dataset collected in Week 1 and the ground truth masks generated this week.

Experimented with preprocessing and normalization to improve flood feature extraction accuracy.

🌐 Web Integration:

Began developing a web interface to visualize model results — currently halfway through building the front-end for viewing flood detection outputs interactively.
