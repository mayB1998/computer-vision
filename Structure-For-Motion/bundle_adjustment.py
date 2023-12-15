import numpy as np
import cv2
from scipy.optimize import least_squares


# Calculation for Reprojection error in the main pipeline
def compute_reprojection_error(world_points, image_points, camera_pose, intrinsic_matrix, homogeneity):
    """
    Compute the reprojection error of 3D world points onto 2D image points.

    Parameters:
    - world_points: 3D coordinates of world points.
    - image_points: 2D coordinates of corresponding image points.
    - camera_pose: Camera pose matrix (extrinsic parameters).
    - intrinsic_matrix: Intrinsic camera matrix.
    - homogeneity: Flag indicating whether to use homogeneous coordinates.

    Returns:
    - total_error: Total reprojection error.
    - transformed_world_points: Transformed world points.
    - projected_points: Projected points on the image.
    """
    total_error = 0

    # Extract rotation and translation from the camera pose matrix
    rotation_matrix = camera_pose[:3, :3]
    translation_vector = camera_pose[:3, 3]

    # Convert rotation matrix to Rodrigues representation
    rvec, _ = cv2.Rodrigues(rotation_matrix)

    # Convert 3D world points to homogeneous coordinates if required
    if homogeneity == 1:
        world_points = cv2.convertPointsFromHomogeneous(world_points.T)

    # Project 3D world points onto the 2D image plane
    projected_points, _ = cv2.projectPoints(world_points, rvec, translation_vector, intrinsic_matrix, distCoeffs=None)
    projected_points = projected_points[:, 0, :]
    projected_points = np.float32(projected_points)

    # Compute reprojection error
    if homogeneity == 1:
        total_error = cv2.norm(projected_points, image_points.T, cv2.NORM_L2)
    else:
        total_error = cv2.norm(projected_points, image_points, cv2.NORM_L2)

    # Transpose image_points for further processing
    image_points = image_points.T

    # Normalize total error by the number of points
    normalized_error = total_error / len(projected_points)

    return normalized_error, world_points, projected_points



# Calculation of reprojection error for bundle adjustment
def calculate_optimization_reprojection_error(params):
    # Extracting parameters
    projection_matrix = params[0:12].reshape((3, 4))
    intrinsic_matrix = params[12:21].reshape((3, 3))
    
    # Extracting image points
    num_image_points = len(params[21:])
    num_image_points_to_use = int(num_image_points * 0.4)
    image_points = params[21:21 + num_image_points_to_use].reshape((2, int(num_image_points_to_use / 2)))
    
    # Extracting 3D points
    num_3d_points = int(len(params[21 + num_image_points_to_use:]) / 3)
    three_d_points = params[21 + num_image_points_to_use:].reshape((num_3d_points, 3))
    
    # Extracting rotation and translation
    rotation_matrix = projection_matrix[:3, :3]
    translation_vector = projection_matrix[:3, 3]

    image_points = image_points.T
    num_points = len(image_points)
    error = []
    
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    
    projected_2d_points, _ = cv2.projectPoints(three_d_points, rotation_vector, translation_vector, intrinsic_matrix, distCoeffs=None)
    projected_2d_points = projected_2d_points[:, 0, :]
    
    for idx in range(num_points):
        actual_image_point = image_points[idx]
        reprojected_image_point = projected_2d_points[idx]
        point_error = (actual_image_point - reprojected_image_point) ** 2
        error.append(point_error)

    error_array = np.array(error).ravel() / num_points

    return error_array

def bundle_adjustment(points_3d, image_points, projection_matrix, intrinsic_matrix, reprojection_error_threshold):
    # Concatenate parameters for optimization
    optimization_variables = np.hstack((projection_matrix.ravel(), intrinsic_matrix.ravel()))
    optimization_variables = np.hstack((optimization_variables, image_points.ravel()))
    optimization_variables = np.hstack((optimization_variables, points_3d.ravel()))

    # Calculate initial error
    initial_error = np.sum(calculate_optimization_reprojection_error(optimization_variables))

    # Use least squares optimization to minimize reprojection error
    optimized_values = least_squares(fun=calculate_optimization_reprojection_error, x0=optimization_variables, gtol=reprojection_error_threshold)

    # Extract optimized values
    optimized_values = optimized_values.x
    optimized_projection_matrix = optimized_values[:12].reshape((3, 4))
    optimized_intrinsic_matrix = optimized_values[12:21].reshape((3, 3))

    remaining_elements = len(optimized_values[21:])
    num_image_points = int(remaining_elements * 0.4)
    optimized_image_points = optimized_values[21:21 + num_image_points].reshape((2, num_image_points // 2)).T

    optimized_world_coordinates = optimized_values[21 + num_image_points:].reshape((int(remaining_elements / 3), 3))
    optimized_image_points = optimized_image_points.T

    return optimized_world_coordinates, optimized_image_points, optimized_projection_matrix