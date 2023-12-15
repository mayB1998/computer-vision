import cv2

def PnP(world_points, image_points, intrinsic_matrix, distortion_coefficients, original_image_points, initial):
    """
    Estimate camera pose using solvePnPRansac.

    Parameters:
    - world_points: 3D coordinates of world points.
    - image_points: 2D coordinates of corresponding image points.
    - intrinsic_matrix: Intrinsic camera matrix.
    - distortion_coefficients: Distortion coefficients.
    - original_image_points: Original 2D image points.
    - initial: Flag indicating whether the function is used in the initial stage.

    Returns:
    - rotation_matrix: Rotation matrix.
    - translation_vector: Translation vector.
    - filtered_image_points: Filtered 2D image points.
    - filtered_world_points: Filtered 3D world points.
    - filtered_original_image_points: Filtered original 2D image points.
    """
    # Check if the function is used in the initial stage
    if initial == 1:
        # If initial, reshape the input arrays for consistency
        world_points = world_points[:, 0, :]  # to remove empty dimension
        image_points = image_points.T
        original_image_points = original_image_points.T

    # Check if there are enough 3D points for PnP
    if len(world_points) < 4:
        print("Not enough points for PnP.")
        return None, None, None, world_points, original_image_points

    # Use solvePnPRansac to estimate the pose (R, t) from 3D-2D correspondences
    ret, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        world_points, image_points, intrinsic_matrix, distortion_coefficients, cv2.SOLVEPNP_ITERATIVE
    )

    # Check if the PnP solution is successful
    if ret is not None:
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # If inliers are obtained, filter the input arrays accordingly
        if inliers is not None:
            image_points = image_points[inliers[:, 0]]
            world_points = world_points[inliers[:, 0]]
            original_image_points = original_image_points[inliers[:, 0]]

        # Return the rotation matrix (R), translation vector (t), filtered 2D points (image_points),
        # filtered 3D points (world_points), and filtered original 2D points (original_image_points)
        return rotation_matrix, translation_vector, image_points, world_points, original_image_points
    else:
        print("PnP failed to find a solution.")
        # If PnP fails, return None for rotation matrix, translation vector, and filtered arrays
        return None, None, None, world_points, original_image_points


