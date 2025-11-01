# Flood-Detection-and-Risk-Perdiction-Using-SAR
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

🗓️ Weekly Progress Report

This week, I explored information about major floods in India and identified the regions most affected. I also researched when these floods typically occur — analyzing seasonal patterns and times of the year when flooding is most frequent.

I examined the SAR satellite images in the collected dataset to understand the differences between pre-flood and post-flood conditions. Additionally, I searched for suitable models to clean and preprocess the data, which will be used for training in the upcoming week.

To support this, I developed two Python scripts:

unet.py – Labels the data based on the time and location of each flood event.

train_flood_unet.py – Creates the dataset and trains the U-Net model for flood prediction.

Next week, the focus will be on refining the dataset and training the model to evaluate its performance.
