import numpy as np
import cv2

def Triangulation(P1, P2, pts1, pts2):
    """
    Perform triangulation to reconstruct 3D points from corresponding 2D points in two images.

    Parameters:
    - P1: Projection matrix for the first camera.
    - P2: Projection matrix for the second camera.
    - pts1: 2D points in the first image.
    - pts2: 2D points in the second image.

    Returns:
    - points1: Transposed 2D points in the first image.
    - points2: Transposed 2D points in the second image.
    - cloud: Triangulated 3D points.
    """
    # Transpose 2D point arrays to get input data in the expected format for cv2.triangulatePoints
    points1 = np.transpose(pts1)
    points2 = np.transpose(pts2)

    # Perform triangulation to obtain homogeneous 3D coordinates
    cloud = cv2.triangulatePoints(P1, P2, points1, points2)
    
    # Convert homogeneous coordinates to non-homogeneous coordinates
    cloud = cloud / cloud[3]

    return points1, points2, cloud
