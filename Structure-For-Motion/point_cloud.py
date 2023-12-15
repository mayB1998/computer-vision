import numpy as np
import open3d as o3d
import copy
import os

def create_pointcloud(output_path, point_cloud, colors):
    """
    Convert 3D point cloud data to PLY format and save it to a file.

    Parameters:
    - output_path: Path to the directory where the PLY file will be saved.
    - point_cloud: 3D point cloud data (numpy array).
    - colors: RGB colors corresponding to each point in the cloud.

    Returns:
    None
    """
    # Rescale point cloud coordinates and concatenate with colors
    scaled_points = point_cloud.reshape(-1, 3) * 200
    colored_points = np.hstack([scaled_points, colors.reshape(-1, 3)])

    # Clean the point cloud by removing points outside a certain distance from the mean
    mean_point = np.mean(colored_points[:, :3], axis=0)
    centered_points = colored_points[:, :3] - mean_point
    distances = np.sqrt(centered_points[:, 0] ** 2 + centered_points[:, 1] ** 2 + centered_points[:, 2] ** 2)
    valid_indices = np.where(distances < np.mean(distances) + 300)
    cleaned_points = colored_points[valid_indices]

    # Define PLY header
    ply_header = '''ply
        format ascii 1.0
        element vertex %(num_verts)d
        property float x
        property float y
        property float z
        property uchar blue
        property uchar green
        property uchar red
        end_header
        '''

    # Write the PLY file
    with open(os.path.join(output_path, "Point_Cloud", "sparse_ptcloud.ply"), "w") as file:
        file.write(ply_header % dict(num_verts=len(cleaned_points)))
        np.savetxt(file, cleaned_points, '%f %f %f %d %d %d')


# Camera pose registration
def camera_orientation(path, mesh, R_T, i):
    """
    Visualize the camera pose and save it to a PLY file.

    Parameters:
    - path: Path to the directory where the PLY file will be saved.
    - mesh: 3D mesh data.
    - R_T: Transformation matrix representing camera pose.
    - i: Index or identifier for the camera.

    Returns:
    None
    """
    T = np.zeros((4, 4))
    T[:3, :] = R_T
    T[3, :] = np.array([0, 0, 0, 1])
    new_mesh = copy.deepcopy(mesh).transform(T)
    o3d.io.write_triangle_mesh(path + "/Point_Cloud/cam_positions/camerapose" + str(i) + '.ply', new_mesh)

