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

🎯 Objective

To build an AI-based system capable of learning from previous flood events and predicting areas at high risk of flooding, supporting early warning and disaster management efforts.
