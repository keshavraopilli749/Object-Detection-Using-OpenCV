# Object Detection Using OpenCV

This project implements real-time object detection using OpenCV with a pre-trained MobileNet SSD model. It detects objects from a live webcam feed, draws bounding boxes around them, and labels each with a confidence score. Techniques like non-maximum suppression (NMS) are employed to avoid multiple detections of the same object.

## Features:
- Real-time object detection from webcam feed
- Detection of multiple objects using the MobileNet SSD model
- Non-maximum suppression to enhance detection accuracy
- Supports various object classes from the COCO dataset

## Prerequisites:
- **Python 3.x**
- **OpenCV**: For computer vision tasks
- **NumPy**: For numerical operations

## Setup:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/keshavraopilli749/Object-Detection-Using-OpenCV.git
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python numpy
   ```

3. **Set up the environment:**  
   (Optional: If you're using Anaconda, create the conda environment)
   ```bash
   conda env create -f objectdetectioncv.yml
   ```

4. **Navigate to the source folder:**
   ```bash
   cd path_to_your_project/source
   ```

5. **Run the main webcam detection script:**
   ```bash
   python main_webcam.py
   ```

## Project Files:
- **coco.names**: Contains the names of detectable objects.
- **frozen_inference_graph.pb**: Pre-trained weights file for the object detection model.
- **ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt**: Config file for the MobileNet SSD model.

## Demo Video:
Watch the following video for a quick demonstration of the object detection process:

[![Object Detection Demo](https://img.youtube.com/vi/luWCJ2pWmhM/0.jpg)](https://www.youtube.com/watch?v=luWCJ2pWmhM)

## Author:
**Keshav Rao**  
[LinkedIn Profile](https://www.linkedin.com/in/keshav-rao-pilli-a18101337)  
[GitHub Repository](https://github.com/keshavraopilli749/Object-Detection-Using-OpenCV)

---

