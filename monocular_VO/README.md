# Monocular Visual Odometry

![map](map.png)

## Overview
This repository contains a Python implementation of monocular visual odometry using FAST feature detection, Lucas-Kanade feature tracking, and five-point motion estimation. The code is designed to process image sequences and estimate the camera's trajectory over time.

## Requirements
- Python 2.7
- Numpy
- OpenCV

## Dataset
Download the [KITTI odometry dataset (grayscale, 22 GB)](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) for testing the visual odometry code.

## Usage
1. Modify the path in `test.py` to point to your image sequences and ground truth trajectories.
2. Run the following command in the terminal:

    ```bash
    python test.py
    ```

## Code Structure

### `visual_odometry.py`
This file contains the main visual odometry implementation, including feature tracking, camera model, and the visual odometry class.

#### Functions
- `featureTracking`: Performs Lucas-Kanade feature tracking between two frames.
- `PinholeCamera`: Represents the camera model with intrinsic parameters.
- `VisualOdometry`: Manages the visual odometry process, including frame processing and motion estimation.

### `test.py`
This file demonstrates the usage of the visual odometry code on the KITTI odometry dataset.

#### Code Snippet
```python
# Example usage in test.py
import numpy as np 
import cv2
from visual_odometry import PinholeCamera, VisualOdometry

# ... (camera and dataset setup)

# Main loop for processing frames
for img_id in range(4541):
    img = cv2.imread('/path/to/your/dataset/' + str(img_id).zfill(6) + '.png', 0)
    vo.update(img, img_id)

    # ... (visualization and trajectory plotting)

cv2.imwrite('map.png', traj)
```

### References
1. [一个简单的视觉里程计实现 | 冯兵的博客](http://fengbing.net/2015/07/26/%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%E8%A7%86%E8%A7%89%E9%87%8C%E7%A8%8B%E8%AE%A1%E5%AE%9E%E7%8E%B01/ )<br>
2. [Monocular Visual Odometry using OpenCV](http://avisingh599.github.io/vision/monocular-vo/) and its related project report [_Monocular Visual Odometry_](http://avisingh599.github.io/assets/ugp2-report.pdf) | Avi Singh
 
Search "cv2.findEssentialMat", "cv2.recoverPose" etc. in github, you'll find more python projects on slam / visual odometry / 3d reconstruction
