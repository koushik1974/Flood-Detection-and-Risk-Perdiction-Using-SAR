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

🗓️ Weekly Progress Report - Week - 1

This week, I explored information about major floods in India and identified the regions most affected. I also researched when these floods typically occur — analyzing seasonal patterns and times of the year when flooding is most frequent.

I examined the SAR satellite images in the collected dataset to understand the differences between pre-flood and post-flood conditions. Additionally, I searched for suitable models to clean and preprocess the data, which will be used for training in the upcoming week.

I used the Copernicus Open Access Hub

- I used the Copernicus Open Access Hub, a platform that provides satellite imagery captured by Sentinel-1 SAR (Synthetic Aperture Radar). This satellite uses deep-penetrating radar waves capable of passing through clouds and capturing accurate reflections of the earth’s surface, including water bodies and surrounding landscapes. This feature makes it ideal for obtaining pre- and post-flood images even under cloudy or stormy conditions.


<img width="1521" height="840" alt="Screenshot 2025-11-08 154554" src="https://github.com/user-attachments/assets/2ebf2216-c68b-4515-a9a8-83b23a4edbcb" />
Screenshot of the interface of Copernicus 


To support this, I developed two Python scripts:

unet.py – Labels the data based on the time and location of each flood event.

train_flood_unet.py – Creates the dataset and trains the U-Net model for flood prediction.

Next week, the focus will be on refining the dataset and training the model to evaluate its performance.

<img width="1081" height="674" alt="Screenshot 2025-11-08 154613" src="https://github.com/user-attachments/assets/a3a8b62e-803c-4a1b-9131-4ab13c68ca8b" />


<img width="1092" height="653" alt="Screenshot 2025-11-08 154623" src="https://github.com/user-attachments/assets/efb97f9a-46cb-4940-9a5c-836c27c0a1ef" />


🗓️ WEEK 2 Progress 

— Ground Truth Generation & Model Training

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


<img width="1328" height="298" alt="Screenshot 2025-11-08 154636" src="https://github.com/user-attachments/assets/094dae51-6703-44b2-ad43-cefa9ac11841" />


🧩 Model Training:

Started training the flood detection model using the dataset collected in Week 1 and the ground truth masks generated this week.

Experimented with preprocessing and normalization to improve flood feature extraction accuracy.

🌐 Web Integration:

Began developing a web interface to visualize model results — currently halfway through building the front-end for viewing flood detection outputs interactively.


🗓️ WEEK 3 Progress
— Web Application Development & Model Integration

In Week 3, the focus is on building an interactive web application that connects the trained U-Net flood detection model with a user-friendly interface. The goal was to allow users to upload SAR image pairs, visualize flood predictions, and receive confidence scores, flood coverage, and regional risk levels directly from the browser.

🌐 Webpage Creation & Front-End Development

This week, I built the first working version of the Flood Detection Web Platform, designed to deliver an intuitive workflow:

✔️ Key Features Developed:

Image Upload Interface:
Created a responsive web page using HTML, CSS, and JavaScript that allows users to upload pre-flood and post-flood SAR image tiles.

Live Preview:
The selected images are displayed instantly on the webpage before processing.

Interactive Visualization Output Section:
Designed placeholder regions for displaying:

Predicted flood mask

Overlay visualization

Confidence score

Flood coverage percentage

Final risk rating (Low / Medium / High)

<img width="1342" height="785" alt="image" src="https://github.com/user-attachments/assets/de4a51d3-7ac9-4383-955e-b3f596ae7086" />


✔️ Design Highlights:

Clean dashboard with dedicated sections for input images and model outputs

Loading animation while the backend model generates predictions

Error handling pop-ups if incorrect or missing images are uploaded

🔗 Backend Integration (FastAPI/Flask)

To connect the webpage with the trained U-Net model, I built a backend API.

✔️ API Endpoints Implemented:
/predict endpoint

ACCEPTS:

Pre-flood SAR image

Post-flood SAR image

Backend Steps:

Preprocesses the uploaded images (resize, normalize).

Feeds them into the trained U-Net model.

GENERATES:

Predicted flood mask

Flood overlay visualization (mask on top of post-flood image)

COMPUTES:

- Model confidence score

- Flood coverage percentage

- Risk level based on thresholded flood area

- Returns the results as JSON + visualization images (PNG).

<img width="1387" height="466" alt="image" src="https://github.com/user-attachments/assets/652f4a35-8c0e-45e1-ade8-8c8e5646ea2c" />


<img width="1397" height="781" alt="image" src="https://github.com/user-attachments/assets/7a57149e-1f34-4405-b069-f3c3c365e95a" />



This is a small thing i started working on after the main project got to finale .

/weather endpoint (Quick Weather Prediction)
- Built a small integration using a weather API to get:
- Current rainfall
- Humidity
- River overflow indicators
- Short-term flood risk hints
- This adds real-time environmental context at the start of every prediction cycle.

🧠 Backend Processing Logic Implemented
✔️ Flood Coverage Calculation

Number of pixels labeled as “flooded” ÷ Total image pixels = Flood Coverage %

✔️ Confidence Score

Softmax output of the segmentation mask averaged across predicted flood-region pixels.

✔️ Risk Level Classification

Based on flood coverage + intensity differences:

- Flood %	Risk Level
- 0–20%	Low
- 20–50%	Medium
- >50%	High
📡 Full Integration Workflow

When the webpage is run:

i) User uploads pre-flood & post-flood images.

ii) The frontend sends them to the backend /predict API.

iii) The U-Net model processes both images.

iv) Backend returns:

v) Flood mask

vi) Flood overlay

vii) Confidence score

viii) Flood coverage

ix) Risk level

The webpage auto-updates the UI with:

- Visual flood map

- Color-coded risk badge

- Numeric analysis (coverage %, confidence)



🖥️ Visual Output Example (as displayed on the webpage)

- Predicted Flood Mask: grayscale / binary mask

- Overlay Visualization: mask placed on SAR post-flood image

- Confidence Score: e.g., 0.87

- Flood Coverage: e.g., 34.2%

- Risk Level: “Medium Risk”

<img width="1400" height="732" alt="image" src="https://github.com/user-attachments/assets/178d5fcb-05cf-4f3c-988d-ab3025770805" />

