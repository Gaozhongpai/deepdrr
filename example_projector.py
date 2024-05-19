#! python3
"""Minimal projection example with DeepDRR."""

import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
from pathlib import Path
import os 
import numpy as np
import json
from plyfile import PlyData, PlyElement
from skimage import measure


os.environ['PYOPENGL_PLATFORM'] = 'osmesa'


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

C0 = 0.28209479177387814
def SH2RGB(sh):
    return sh * C0 + 0.5

    # Function to sample points based on volume intensity
def sample_points(volume, num_points=5000):
    flattened_volume = volume.flatten()
    probabilities = flattened_volume / np.sum(flattened_volume)
    chosen_indices = np.random.choice(a=len(probabilities), size=num_points, p=probabilities)
    return np.column_stack(np.unravel_index(chosen_indices, volume.shape))

def generate_random_point_cloud(verts, num_points=25000):
    # Calculate the minimum and maximum bounds for each axis
    min_bounds = np.min(verts, axis=0)
    max_bounds = np.max(verts, axis=0)
    
    # Generate random points within these bounds
    random_points = np.random.uniform(low=min_bounds, high=max_bounds, size=(num_points, 3))
    
    return random_points

def generate_evenly_sampled_point_cloud(verts, num_points=25000):
    # Calculate the minimum and maximum bounds for each axis
    min_bounds = np.min(verts, axis=0)
    max_bounds = np.max(verts, axis=0)
    
    # Calculate the total number of points in each dimension needed to get approximately num_points total
    total_points = num_points
    num_points_per_dim = int(np.ceil(total_points ** (1/3)))
    
    # Generate evenly spaced points within each dimension
    x_points = np.linspace(min_bounds[0], max_bounds[0], num_points_per_dim)
    y_points = np.linspace(min_bounds[1], max_bounds[1], num_points_per_dim)
    z_points = np.linspace(min_bounds[2], max_bounds[2], num_points_per_dim)
    
    # Create a meshgrid and reshape to a list of points
    xx, yy, zz = np.meshgrid(x_points, y_points, z_points)
    grid_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    
    # If the number of grid points is greater than required, randomly sample num_points from the grid
    if len(grid_points) > num_points:
        indices = np.random.choice(len(grid_points), num_points, replace=False)
        sampled_points = grid_points[indices]
    else:
        sampled_points = grid_points
    
    return sampled_points

def randomly_sample_point_cloud(point_cloud, num_samples=15000):
    # Check if the point cloud has fewer points than the number of samples requested
    if len(point_cloud) <= num_samples:
        return point_cloud

    # Randomly select `num_samples` indices
    sampled_indices = np.random.choice(len(point_cloud), num_samples, replace=False)

    # Select the points at these indices
    sampled_points = point_cloud[sampled_indices]

    return sampled_points


# Define the permutation matrix that swaps the x and z axes
P_xz = np.array([
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])
# Define the permutation matrix that swaps the x and y axes
P_xy = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

# Define the permutation matrix that swaps the y and z axes
P_yz = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

P_120 = np.array([ ## 
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])



def main(root, filename):    
    filepath = os.path.join(root, filename)
    ct = deepdrr.Volume.from_nifti(filepath, use_thresholding=True)
    ct.supine()

    output_dir = Path("/code/dataset/deepdrr/{}".format(Path(filename).with_suffix('').stem))
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path_train = os.path.join(output_dir, 'transforms_train.json')
    file_path_test = os.path.join(output_dir, 'transforms_test.json')
    
    # Combined rotation: Rotate 90 degrees around the x-axis and -90 degrees around the z-axis
    # Step-by-step combined transformations:
    # 1. np.transpose(volume, (2, 0, 1)) swaps the axes to [Z, X, Y]
    # 2. [::-1, :, :] flips the new first dimension (Z-axis), effectively rotating -90 degrees around z-axis
    # 3. [:, :, ::-1] flips the new third dimension (Y-axis), effectively rotating 90 degrees around x-axis
    volume = np.transpose(ct.data, (2, 0, 1))[::-1, :, :][:, :, ::-1]
    shape = np.array(volume.shape)
    spacing = ct.spacing.data[[2, 0, 1]] # [Z, X, Y]
    print("spacing: {}".format(spacing))
    
    binary_volume = volume > 1.25
    
    # Step 2: Surface Extraction
    verts, faces, _, _ = measure.marching_cubes(binary_volume, level=0, step_size=3, spacing=spacing[:3])
    # center_to_world = ct.center_in_world.data[[1, 2, 0]]
    verts = randomly_sample_point_cloud(verts)
    
    verts = verts - shape / 2 * spacing[:3] + ct.center_in_world.data[:3]
    
    shs = np.random.random((verts.shape[0], 3)) / 255.0
    # pcd = BasicPointCloud(points=verts, colors=SH2RGB(shs), normals=np.zeros((verts.shape[0], 3)))
    # storePly(os.path.join(output_dir, "points3d_mc.ply"), verts, SH2RGB(shs) * 255)

    # Generate point cloud
    sparse = (sample_points(volume, num_points=5000) - shape / 2) * spacing[:3] + ct.center_in_world.data[:3]
    shs = np.random.random((sparse.shape[0], 3)) / 255.0
    # storePly(os.path.join(output_dir, "points3d_sparse.ply"), sparse, SH2RGB(shs) * 255)

    # We create random points inside the bounds of the synthetic Blender scenes
    verts = np.concatenate((verts, sparse))
    shs = np.random.random((verts.shape[0], 3)) / 255.0
    # pcd = BasicPointCloud(points=verts, colors=SH2RGB(shs), normals=np.zeros((verts.shape[0], 3)))
    # storePly(os.path.join(output_dir, "points3d.ply"), verts, SH2RGB(shs) * 255)
    
    vert_3dgs = generate_random_point_cloud(verts)
    shs = np.random.random((vert_3dgs.shape[0], 3)) / 255.0
    # pcd = BasicPointCloud(points=verts, colors=SH2RGB(shs), normals=np.zeros((verts.shape[0], 3)))
    storePly(os.path.join(output_dir, "points3d_3dgs.ply"), vert_3dgs, SH2RGB(shs) * 255)
    
    vert_x = generate_evenly_sampled_point_cloud(verts)
    shs = np.random.random((vert_x.shape[0], 3)) / 255.0
    # pcd = BasicPointCloud(points=verts, colors=SH2RGB(shs), normals=np.zeros((verts.shape[0], 3)))
    storePly(os.path.join(output_dir, "points3d_x.ply"), vert_x, SH2RGB(shs) * 255)
    
    # # define the simulated C-arm
    # carm = deepdrr.device.SimpleDevice()
    # print("sensor_height: {}, sensor_weight: {}".format(carm.sensor_height, carm.sensor_width))
    # print("pixel size: {}".format(carm.pixel_size))
    # print("source_to_detector_distance: {}".format(carm.source_to_detector_distance))
    # print("intrinsic:{}".format(carm.camera_intrinsics.data))
    # # Define the range of angles (in degrees) to sample
    # angles_train = np.linspace(-60, 60, 60)  # 10 samples from 0 to 360 degrees
    # os.makedirs(output_dir / "train", exist_ok=True)
    # angles_test = np.random.randint(-60, 60, 50)  # 10 samples from 0 to 360 degrees
    # os.makedirs(output_dir / "test", exist_ok=True)
    # # Loop through each angle and project
    # with Projector(ct, device=carm, intensity_upper_bound=4) as projector:
    #     shape = projector.output_shape
    #     print("shape: {}".format(shape))
    #     print("source_to_detector_distance: {}".format(projector.source_to_detector_distance))
        
    #     camera_angle_x_rad = 2 * np.arctan(0.5 * shape[0] / projector.source_to_detector_distance)
    #     json_data_train = {"camera_angle_x": camera_angle_x_rad, "frames": []}
    #     json_data_test = {"camera_angle_x": camera_angle_x_rad, "frames": []}
            
    #     for idx, angle in enumerate(angles_train):
    #         p = ct.center_in_world
    #         # Create a rotation matrix for the current angle around the z-axis
    #         rotation = geo.Rotation.from_euler('z', np.radians(angle))
    #         # Apply the rotation to the initial view direction (convert to 3D first)
    #         initial_direction = np.array([0, 1, 0])
    #         rotated_direction = rotation.apply(initial_direction)
    #         v = ct.world_from_anatomical @ geo.vector(rotated_direction)
    #         # Set the view
    #         carm.set_view(
    #             p,
    #             v,
    #             up=ct.world_from_anatomical @ geo.vector(0, 0, 1),
    #             source_to_point_fraction=0.7,
    #         )
            
    #         T = carm.world_from_camera3d.data

    #         # # permute [1, 2, 0]
    #         # T = P_120 @ T
    #         # # Flip the direction of the x-axis
    #         # T[:3, 0] *= -1
    #         # Flip the direction of the z-axis
    #         T[:3, 1:3] *= -1
    #         frame_data = {"file_path": "./train/{0:0=3d}".format(idx),
    #                     "transform_matrix": T.tolist()
    #             }
    #         json_data_train['frames'].append(frame_data)
            
    #         # Project and save the image
    #         image = projector()
    #         path = output_dir / "train" /  f"{idx:03d}.png"
    #         image_utils.save(path, image)
    #         print(f"Saved train projection image to {path.absolute()}")
            
    #     # Loop through each angle and project
    #     for idx, angle in enumerate(angles_test):
    #         p = ct.center_in_world
    #         # Create a rotation matrix for the current angle around the z-axis
    #         rotation = geo.Rotation.from_euler('z', np.radians(angle))
    #         # Apply the rotation to the initial view direction (convert to 3D first)
    #         initial_direction = np.array([0, 1, 0])
    #         rotated_direction = rotation.apply(initial_direction)
    #         v = ct.world_from_anatomical @ geo.vector(rotated_direction)
    #         # Set the view
    #         carm.set_view(
    #             p,
    #             v,
    #             up=ct.world_from_anatomical @ geo.vector(0, 0, 1),
    #             source_to_point_fraction=0.7,
    #         )
            
    #         T = carm.world_from_camera3d.data
    #         # # permute [1, 2, 0]
    #         # T = P_120 @ T
    #         # # Flip the direction of the x-axis
    #         # T[:3, 0] *= -1
    #         # Flip the direction of the z-axis
    #         T[:3, 1:3] *= -1
    #         frame_data = {"file_path": "./test/{0:0=3d}".format(idx),
    #                     "transform_matrix": T.tolist()
    #             }
    #         json_data_test['frames'].append(frame_data)
            
    #         # Project and save the image
    #         image = projector()
    #         path = output_dir / "test" / f"{idx:03d}.png"
    #         image_utils.save(path, image)
            
    #         print(f"Saved test projection image to {path.absolute()}")

    # with open(file_path_train, 'w') as file:
    #     json.dump(json_data_train, file, indent=4)
    # with open(file_path_test, 'w') as file:
    #     json.dump(json_data_test, file, indent=4)

if __name__ == "__main__":
    
    root = "/code/dataset/CTPelvic1K_dataset6_data"
    files = os.listdir(root)
    # Filter files that end with .nii.gz
    nii_gz_files = [file for file in files if file.endswith('.nii.gz')]
    nii_gz_files.sort()
    for filename in nii_gz_files[:10]:
        print(filename)
        main(root, filename)
