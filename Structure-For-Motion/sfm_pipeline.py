# sfm_pipeline.py
import os
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from PnP import *
from triangulation import *
from feature_detection import *
from bundle_adjustment import *
from point_cloud import *
import plotly.graph_objects as go

fig = go.Figure()


def draw_points(image, pts, repro):
    if repro == False:
        image = cv2.drawKeypoints(image, pts, image, color=(0, 255, 0), flags=0)
    else:
        for p in pts:
            image = cv2.circle(image, tuple(p), 2, (0, 0, 255), -1)
    return image

def sfm_pipeline(path, img_dir, intrinsic_matrix, perform_bundle_adjustment=False):
    image_dataset = os.path.basename(img_dir)

    # Initialize variables and parameters
    print('Initialized Initial Parameters')
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    pose_array = intrinsic_matrix.ravel()

    # Initialize Pose at a (4 x 4) Identity Matrix
    initial_pose_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])                
    current_pose_matrix = np.empty((3, 4))

    projection_matrix_1 = np.matmul(intrinsic_matrix, initial_pose_matrix)
    reference_projection_matrix = projection_matrix_1
    projection_matrix_2 = np.empty((3, 4))
    total_points_3d = np.zeros((1, 3))
    total_colors = np.zeros((1, 3))
    images = []

    # Get a list of images in the directory
    for img_filename in sorted(os.listdir(img_dir)):
        if '.jpg' in img_filename.lower() or '.png' in img_filename.lower():
            images += [img_filename]

    i = 0
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()

    # Setting the reference two frames
    image_0 = cv2.imread(os.path.join(img_dir, images[i]))
    image_1 = cv2.imread(os.path.join(img_dir, images[i + 1]))

    # Initial features for the first two images
    pts0, pts1 = find_features(image_0, image_1)

    # Finding essential matrix
    essential_matrix, mask = cv2.findEssentialMat(pts0, pts1, intrinsic_matrix, method=cv2.RANSAC, prob=0.999, threshold=0.4, mask=None)
    pts0 = pts0[mask.ravel() == 1]
    pts1 = pts1[mask.ravel() == 1]

    # The pose obtained is for the second image with respect to the first image
    _, rotation_matrix, translation, mask = cv2.recoverPose(essential_matrix, pts0, pts1, intrinsic_matrix)  # finding the pose
    pts0 = pts0[mask.ravel() > 0]
    pts1 = pts1[mask.ravel() > 0]

    # Initial extrinsic matrix for the second image
    current_pose_matrix[:3, :3] = np.matmul(rotation_matrix, initial_pose_matrix[:3, :3])
    current_pose_matrix[:3, 3] = initial_pose_matrix[:3, 3] + np.matmul(initial_pose_matrix[:3, :3], translation.ravel())
    
    # Initial Projection Matrix for the second image
    projection_matrix_2 = np.matmul(intrinsic_matrix, current_pose_matrix)

    # Triangulate the first image pair and poses will be set as reference
    pts0, pts1, points_3d = Triangulation(projection_matrix_1, projection_matrix_2, pts0, pts1)

    # Reprojecting the 3D points on the image and calculating the reprojection error
    reprojection_error, points_3d, reprojection_points = compute_reprojection_error(points_3d, pts1, current_pose_matrix, intrinsic_matrix, homogeneity=1)
    print("REPROJECTION ERROR: ", reprojection_error)
    rotation, translation, pts1, points_3d, keypoints_0_transformed = PnP(points_3d, pts1, intrinsic_matrix, np.zeros((5, 1), dtype=np.float32), pts0, initial=1)

    rotation = np.eye(3)
    translation = np.array([[0], [0], [0]], dtype=np.float32)

    # Images to be taken into consideration
    total_images = len(images) - 2 

    # Store the projection matrices for the current image pair in the pose_array array
    pose_array = np.hstack((pose_array, projection_matrix_1.ravel()))
    pose_array = np.hstack((pose_array, projection_matrix_2.ravel()))

    # Gradient threshold for bundle adjustment optimization
    gradient_threshold = 0.4


    for image_index in tqdm(range(total_images)):
        # Acquire a new image to be added to the pipeline and acquire matches with image pair
        current_image = cv2.imread(os.path.join(img_dir, images[image_index + 2]))

        # Detect features in the new image (current_image) and find correspondences
        pts_, pts2  = find_features(image_1, current_image)

        # If not processing the first image pair, perform triangulation for 3D reconstruction
        if image_index != 0:
            # Triangulate 3D points using the previous camera poses (projection_matrix_1, projection_matrix_2) and correspondences (keypoints_previous, keypoints_current)
            pts0, pts1, points_3d = Triangulation(projection_matrix_1, projection_matrix_2, pts0, pts1)

            # Transpose the points matrix for further processing
            pts1 = pts1.T

            # Convert homogeneous coordinates of 3D points
            points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)
            points_3d = points_3d[:, 0, :]

        # Data Association
        # Find common points between the 3D points from the previous pair and the new correspondences
        # Extract common points in keypoints_current and keypoints_previous
        indices_1, indices_2, temp_keypoints_current, temp_keypoints_new = data_association(pts1, pts_, pts2)

        common_keypoints_current_image = pts2[indices_2]
        common_keypoints_previous = pts_[indices_2]

        # We have the 3D - 2D Correspondence for the new image as well as the point cloud obtained from before. The common points can be used to find the world coordinates of the new image
        # using Perspective - n - Point (PnP)
        rotation, translation, common_keypoints_current_image, points_3d, common_keypoints_previous = PnP(points_3d[indices_1], common_keypoints_current_image, intrinsic_matrix, np.zeros((5, 1), dtype=np.float32), common_keypoints_previous, initial=0)

        # Find the equivalent projection matrix for the new image
        new_extrinsic_matrix = np.hstack((rotation, translation))
        new_projection_matrix = np.matmul(intrinsic_matrix, new_extrinsic_matrix)

        error, points_3d, _ = compute_reprojection_error(points_3d, common_keypoints_current_image, new_extrinsic_matrix, intrinsic_matrix, homogeneity=0)

        temp_keypoints_previous, temp_keypoints_current, points_3d = Triangulation(projection_matrix_2, new_projection_matrix, temp_keypoints_current, temp_keypoints_new)
        error, points_3d, _ = compute_reprojection_error(points_3d, temp_keypoints_current, new_extrinsic_matrix, intrinsic_matrix, homogeneity=1)
        print("Reprojection Error: ", error)

        # We are storing the pose for each image. This will be very useful during multiview stereo as this should be known
        pose_array = np.hstack((pose_array, new_projection_matrix.ravel()))

        # Bundle adjustment based on least squares
        if perform_bundle_adjustment:
            print("Bundle Adjustment...")
            points_3d, temp_keypoints_current, new_projection_matrix = bundle_adjustment(points_3d, temp_keypoints_current, new_projection_matrix, intrinsic_matrix, gradient_threshold)
            new_projection_matrix = np.matmul(intrinsic_matrix, new_projection_matrix)
            error, points_3d, _ = compute_reprojection_error(points_3d, temp_keypoints_current, new_extrinsic_matrix, intrinsic_matrix, homogeneity=0)
            print("Minimized error: ", error)
            total_points_3d = np.vstack((total_points_3d, points_3d))
            keypoints_current_reg = np.array(temp_keypoints_current, dtype=np.int32)
            colors = np.array([current_image[l[1], l[0]] for l in keypoints_current_reg])
            total_colors = np.vstack((total_colors, colors))
        else:
            total_points_3d = np.vstack((total_points_3d, points_3d[:, 0, :]))
            keypoints_current_reg = np.array(temp_keypoints_current, dtype=np.int32)
            colors = np.array([current_image[l[1], l[0]] for l in keypoints_current_reg.T])
            total_colors = np.vstack((total_colors, colors)) 
        #camera_orientation(path, mesh, new_projection_matrix, image_index + 2)    

        projection_matrix_1 = np.copy(projection_matrix_2)

        fig.add_trace(go.Scatter(x=[i], y=[error], mode='markers'))
        fig.update_layout(title='Reprojection Error Over Iterations', xaxis_title='Iteration', yaxis_title='Reprojection Error')

        image_0 = np.copy(image_1)
        image_1 = np.copy(current_image)
        pts0 = np.copy(pts_)
        pts1 = np.copy(pts2)
        projection_matrix_2 = np.copy(new_projection_matrix)
        cv2.imshow('image', current_image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    fig.show()
    cv2.destroyAllWindows()

    # the obtained points cloud is registered and saved using open3d. It is saved in .ply form, which can be viewed using meshlab
    print("Processing Point Cloud...")
    create_pointcloud(path, total_points_3d, total_colors)
    print(f"Incremental SFM for {image_dataset} dataset completed!")
    # Saving projection matrices for all the images
    #np.savetxt('pose.csv', posearr, delimiter = '\n')
