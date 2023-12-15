import cv2
import numpy as np

def find_features(img0, img1):
    """
    Detect keypoints and compute descriptors for two images using the SIFT algorithm.

    Parameters:
    - img0: First input image.
    - img1: Second input image.

    Returns:
    - pts0: Keypoints in the first image.
    - pts1: Keypoints in the second image.
    """
    # Convert images to grayscale
    img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    kp0, des0 = sift.detectAndCompute(img0gray, None)
    kp1, des1 = sift.detectAndCompute(img1gray, None)

    # Use Brute-Force Matcher for descriptor matching
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des0, des1, k=2)

    # Apply ratio test to find good matches
    good = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good.append(m)

    # Extract coordinates of keypoints from good matches
    pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good])
    return pts0, pts1


def data_association(pts1, pts2, pts3):
    """
    Perform data association between initial points and new points.

    Find common points between two sets of points (pts1 and pts2) and a third set (pts3).

    Parameters:
    - pts1: Points from the first image.
    - pts2: Points from the second image.
    - pts3: Points from the third image.

    Returns:
    - common_indices_1: Indices of common points in pts1.
    - common_indices_2: Indices of common points in pts2.
    - unmatched_pts2: Points in pts2 that are not common.
    - unmatched_pts3: Points in pts3 that are not common.
    """
    if pts1.shape[0] == 0 or pts2.shape[0] == 0 or pts3.shape[0] == 0:
        print("One or more sets of points are empty.")
        return None, None, None, None

    common_indices_1 = []
    common_indices_2 = []

    # Iterate through points in pts1
    for i in range(pts1.shape[0]):
        matching_indices = np.where((pts2 == pts1[i, :]).all(axis=1))
        # If a match is found in pts2
        if matching_indices[0].size > 0:
            common_indices_1.append(i)
            common_indices_2.append(matching_indices[0][0])

    if not common_indices_1 or not common_indices_2:
        print("No common points found.")
        return None, None, None, None

    # Create masked arrays to filter out common points
    unmatched_pts2 = np.ma.array(pts2, mask=False)
    unmatched_pts2.mask[common_indices_2] = True
    unmatched_pts2 = unmatched_pts2.compressed()
    unmatched_pts2 = unmatched_pts2.reshape(int(unmatched_pts2.shape[0] / 2), 2)

    unmatched_pts3 = np.ma.array(pts3, mask=False)
    unmatched_pts3.mask[common_indices_2] = True
    unmatched_pts3 = unmatched_pts3.compressed()
    unmatched_pts3 = unmatched_pts3.reshape(int(unmatched_pts3.shape[0] / 2), 2)

    return np.array(common_indices_1), np.array(common_indices_2), unmatched_pts2, unmatched_pts3
