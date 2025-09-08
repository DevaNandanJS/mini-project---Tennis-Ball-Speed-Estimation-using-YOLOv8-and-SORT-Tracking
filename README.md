# Tennis Ball Speed Estimation using YOLOv8 and SORT Tracking

Results drive link - https://drive.google.com/drive/folders/14rNjuS5ADS02gaW1RUvC5Ai-rhCujjC5?usp=sharing


This repository contains the code and report for a computer vision mini-project focused on detecting, tracking, and estimating the speed of a tennis ball from video footage. This project implements a complete pipeline from video input to data analysis and visualization.

## Table of Contents
1.  [Project Objective](#project-objective)
2.  [Problem Statement](#problem-statement)
3.  [Our Approach & Pipeline](#our-approach--pipeline)
4.  [Tools and Technologies](#tools-and-technologies)
5.  [Model & Tracker Selection Rationale](#model--tracker-selection-rationale)
6.  [Results](#results)
7.  [Challenges and Limitations](#challenges-and-limitation)
8.  [Future Improvements](#future-improvements)
9.  [Project Structure](#project-structure)
10. [How to Run](#how-to-run)

---

## Project Objective

The primary goal of this project is to develop a robust computer vision pipeline capable of automatically analyzing sports video footage to extract key performance metrics. Specifically, the objective is to accurately estimate the instantaneous and average speed of a moving tennis ball, simulating a real-world sports-tech application used for coaching, broadcasting, and enhancing fan engagement.

---

## Problem Statement

The task is to design and implement a system that can take a video clip of a tennis match as input and perform the following actions:
1.  [cite_start]**Detect** the tennis ball in each frame of the video. [cite: 14]
2.  [cite_start]**Track** the ball's movement across consecutive frames to form a continuous trajectory. [cite: 14]
3.  [cite_start]**Calculate** the approximate speed of the ball in km/h by converting its pixel displacement to real-world distance. [cite: 15]
4.  [cite_start]**Visualize** the results by overlaying the trajectory on the output video and plotting a speed-time graph. [cite: 32, 33]

---

## Our Approach & Pipeline

To solve this problem, I developed a multi-stage pipeline that processes the video sequentially to produce the final analytics. The workflow is as follows:

1.  **Video Input & Pre-processing:** The pipeline begins by loading the input video. Each frame is then pre-processed to enhance detection accuracy. This includes applying **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to improve the ball's visibility against the court and background, especially in varying lighting conditions.

2.  **Object Detection:** A custom-trained **YOLOv8n** model is used to detect the tennis ball in every frame. YOLOv8n was chosen for its excellent balance of speed and accuracy, making it suitable for real-time applications. The model outputs bounding box coordinates for any detected ball.

3.  [cite_start]**Object Tracking:** The detection data is fed into the **SORT (Simple Online and Realtime Tracking)** algorithm. [cite: 22] SORT is responsible for assigning a consistent ID to the detected ball across frames, even if the detector momentarily fails. This allows us to build a continuous and reliable trajectory of the ball's movement.

4.  **Data Filtering and Smoothing:**
    * The raw coordinates from the tracker are first filtered to remove noise and outliers.
    * The trajectory data is then smoothed using interpolation and a **Savitzky-Golay filter**. This step is crucial for calculating a stable and realistic speed, as it minimizes the impact of minor detection jitters.

5.  **Speed Calculation:**
    * [cite_start]A pixel-to-meter conversion ratio is established based on the known width of a tennis court (10.97 meters). [cite: 17, 26]
    * The smoothed pixel coordinates are converted into real-world coordinates (in meters).
    * [cite_start]The distance the ball travels between consecutive frames is calculated, and by using the video's frame rate (FPS), the instantaneous speed is computed (Speed = Distance / Time). [cite: 16, 29]

6.  **Output Generation:** The pipeline produces four key outputs:
    * [cite_start]An **annotated video** showing the detected ball, its bounding box, and a fading trace of its trajectory. [cite: 44]
    * A **CSV file** containing the frame-by-frame coordinates of the ball.
    * A **plot image (.png)** visualizing the ball's trajectory and a graph of its speed over time.
    * [cite_start]**Terminal output** summarizing the key statistics, including average and peak speed. [cite: 45]

Here is a conceptual diagram of the pipeline:

---

## Tools and Technologies

This project leverages a hybrid environment and a combination of powerful open-source libraries:

* **Model Training:**
    * **Google Colab:** Used for training the YOLOv8 model on a custom dataset. The cloud GPU resources available on Colab significantly accelerated the training process.

* **Pipeline Execution & Development:**
    * **Visual Studio Code:** Served as the primary IDE for writing, debugging, and running the main processing script (`track3.py`).

* **Core Libraries:**
    * **Python 3.x:** The primary programming language for the project.
    * **OpenCV:** Used for all video and image processing tasks, such as reading frames, applying CLAHE, and writing the output video.
    * **Ultralytics YOLOv8:** The core object detection framework. I used a fine-tuned `YOLOv8n` model.
    * **Supervision:** A high-level library used for annotating frames with bounding boxes, labels, and traces, simplifying the visualization code.
    * **SORT:** The tracking algorithm used to maintain object identity across frames.
    * **NumPy & Pandas:** Used for efficient numerical operations and data manipulation, especially for handling the coordinate data.
    * **Matplotlib & Scipy:** Used for plotting the final graphs and for signal processing tasks like trajectory smoothing.

---

## Model & Tracker Selection Rationale

* **Why YOLOv8?**
    The YOLO (You Only Look Once) family of models is renowned for its high speed and accuracy in object detection. I chose **YOLOv8n** (the nano version) specifically because it provides a fantastic trade-off between performance and computational cost. For a small, fast-moving object like a tennis ball, having a real-time detector is essential. While larger models like YOLOv8m or YOLOv8l might offer slightly better accuracy, YOLOv8n is lightweight enough to run smoothly without requiring a high-end GPU for inference, making the pipeline more accessible.

* **Why SORT Tracker?**
    For this specific problem, the goal is to track a single, primary object (the ball). [cite_start]The **SORT** algorithm is an excellent choice because it is computationally efficient and highly effective for this scenario. [cite: 22] It primarily uses a Kalman filter to predict the object's location in the next frame and associates detections based on IoU (Intersection over Union). Unlike more complex trackers like DeepSORT, it does not use a deep learning model for re-identification, which makes it much faster and simpler to implement—a perfect fit for a project where speed and a clean implementation are key.

---

## Results

The pipeline was successfully implemented and tested on sample video footage. The system was able to detect and track the tennis ball, calculate its speed, and generate all the specified outputs.

**Final Statistics from a Sample Run:**
