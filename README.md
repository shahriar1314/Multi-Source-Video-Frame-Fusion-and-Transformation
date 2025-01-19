# **Multi-Source Video Frame Fusion and Transformation**

This project implements a solution to fuse multiple sources of image data, such as sequences from moving cameras or videos from different cameras, into a single 2D reference frame. The system processes images, keypoints, and YOLO detections to align all input data to a common reference frame, enabling unified analysis.  

The project avoids dependencies on external computer vision libraries like OpenCV, ensuring compatibility with restricted evaluation environments.  

---

## **Project Overview**
### **Input**
- **Reference Image** (`ref_dir`):
  - Image (`img_ref.jpg`)
  - Keypoints and descriptors (`kp_ref.mat`)

- **Input Videos** (`inputK_dir` for each video K):
  - Sequence of images (`img_0001.jpg`, `img_0002.jpg`, ...)
  - YOLO detections (`yolo_0001.mat`, `yolo_0002.mat`, ...) containing:
    - **Bounding boxes** (`xyxy`): Pixel coordinates for detected objects.
    - **Object IDs** (`id`): Identifiers for detected objects.
    - **Object classes** (`class`): Classes of detected objects.
  - Keypoints and descriptors (`kp_0001.mat`, `kp_0002.mat`, ...).

### **Output**
For each input video:
1. **Homography Matrices**:
   - File: `homographies.mat`  
   - Contains a (3x3xNv) array of homographies, mapping each frame in the video to the reference frame.

2. **Transformed YOLO Detections**:
   - Files: `yolooutput_0001.mat`, `yolooutput_0002.mat`, ...
   - Each file contains the transformed YOLO data in the same format as the input.

---

## **Features**
- Computes homography matrices to map video frames to a reference frame.
- Transforms YOLO bounding boxes to the reference frame.
- Fully implemented in Python without relying on external computer vision libraries like OpenCV.

---

### **Running the Project**
To execute the project, run the `main.py` script with the following arguments:

```bash
python main.py ref_dir input1_dir output1_dir input2_dir output2_dir ... inputN_dir outputN_dir
```
## **Contribuition**
- Sergei Chashnikov
