import itertools
import os
import sys
from typing import NamedTuple

import google.protobuf as pb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import PIL

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

from waymo_open_dataset.utils import box_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils


class FrameInfo:

    def __init__(self, frame: open_dataset.Frame):
        self.src_frame = frame
        self.world_from_frame = np.array(frame.pose.transform).reshape((4, 4))
        self.range_images, self.camera_projections, self.range_image_top_pose = (
            frame_utils.parse_range_image_and_camera_projection(frame))

'''
def extract_polar_and_cartesian_from_range_image(range_image,
                                                 extrinsic,
                                                 inclination,
                                                 pixel_pose=None,
                                                 frame_pose=None,
                                                 dtype=tf.float32,
                                                 scope=None):
    """Adapted from
    `range_image_utils.extract_point_cloud_from_range_image`.
    """
    with tf.compat.v1.name_scope(
        scope, 'ExtractPointCloudFromRangeImage',
        [range_image, extrinsic, inclination, pixel_pose, frame_pose]):
        range_image_polar = range_image_utils.compute_range_image_polar(
            range_image, extrinsic, inclination, dtype=dtype)
        range_image_cartesian = range_image_utils.compute_range_image_cartesian(
            range_image_polar,
            extrinsic,
            pixel_pose=pixel_pose,
            frame_pose=frame_pose,
            dtype=dtype)
    return range_image_polar, range_image_cartesian


def lidar1_project_labels_on_range_image(frame_info: FrameInfo):
    """Adapted from `frame_utils.convert_range_image_to_cartesian`
    """
    frame_pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame_info.src_frame.pose.transform), [4, 4]))

    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=frame_info.range_image_top_pose.data),
        frame_info.range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in frame_info.src_frame.context.laser_calibrations:
        if c.name != open_dataset.LaserName.TOP:  # Only the top laser range image
            continue
        range_image = frame_info.range_images[c.name][0]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        pixel_pose_local = tf.expand_dims(range_image_top_pose_tensor, axis=0)
        frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        polar, cartesian = extract_polar_and_cartesian_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)
        # polar, cartesian = \
        #     tf.cast(polar, dtype=tf.float64), tf.cast(cartesian, dtype=tf.float64)

        # To world frame as per the given `frame_pose`
        # [B, 4, 4]
        range_image_points = cartesian
        world_from_vehicle = tf.cast(frame_pose_local, dtype=range_image_points.dtype)
        world_from_vehicle_rotation = world_from_vehicle[:, 0:3, 0:3]
        world_from_vehicle_translation = world_from_vehicle[:, 0:3, 3]
        # [B, H, W, 3]
        range_image_points = tf.einsum(
          'bij,bhwj->bhwi', world_from_vehicle_rotation,
          range_image_points) + world_from_vehicle_translation[:, tf.newaxis,
                                                             tf.newaxis, :]

        # convert to pixel frame
        pixel_from_world = tf.linalg.inv(pixel_pose_local)
        # [B, H, W, 3, 3]
        pixel_from_world_rotation = pixel_from_world[..., 0:3, 0:3]
        # [B, H, W, 3]
        pixel_from_world_translation = pixel_from_world[..., 0:3, 3]
        # [B, H, W, 3]
        print(pixel_from_world_rotation.shape)
        print(range_image_points.shape)
        range_image_points = tf.einsum(
            'bhwij,bhwj->bhwi', pixel_from_world_rotation,
            range_image_points) + pixel_from_world_translation

        # To sensor frame
        # [B, 3, 3]
        sensor_from_vehicle = tf.linalg.inv(extrinsic)
        sensor_from_vehicle = tf.cast(sensor_from_vehicle, dtype=range_image_points.dtype)
        sensor_from_vehicle = tf.expand_dims(sensor_from_vehicle, axis=0)
        sensor_from_vehicle_rotation = sensor_from_vehicle[..., 0:3, 0:3]
        # translation [B, 1, 3]
        sensor_from_vehicle_translation = tf.expand_dims(
            tf.expand_dims(sensor_from_vehicle[..., 0:3, 3], 1), 1)
        # [B, H, W, 3]
        print(sensor_from_vehicle_rotation.shape)
        print(range_image_points.shape)
        range_image_points = tf.einsum('bkr,bijr->bijk', sensor_from_vehicle_rotation,
                                    range_image_points) + sensor_from_vehicle_translation
'''


def save_ply(points: np.ndarray, out_path: Path, transform=None, format='binary_little_endian') -> None:
    assert points.dtype == np.float32
    assert format in ('ascii', 'binary_little_endian'), "Invalid PLY format!"
    if transform is not None:
        points[:, 3:] = (transform[:3, :3] @ points[:, 3:].T + transform[:3, 3:]).T
    with out_path.open('wb') as f:
        f.write(("ply\n" +
                f"format {format} 1.0\n" +
                f"element vertex {len(points)}\n" +
                 "property float x\n" +
                 "property float y\n" +
                 "property float z\n" +
                 "property float intensity\n" +
                 "property float elongation\n" +
                 "end_header\n").encode('ascii'))
        if format == 'ascii':
            for p in points:
                _range, i, e, x, y, z = p
                f.write(f'{x:.3f} {y:.3f} {z:.3f} {i:.3f} {e:.3f}\n'.encode())
        if format == 'binary_little_endian':
            assert sys.byteorder == 'little'
            f.write(points[:, [3, 4, 5, 1, 2]].tobytes())


def plot_range_image_helper(data, name, layout=None, vmin = 0, vmax=1, cmap='gray'):
    """Plots range image.

    Args:
        data: range image data
        name: the image title
        layout: plt layout
        vmin: minimum value of the passed data
        vmax: maximum value of the passed data
        cmap: color map
    """
    if layout: plt.subplot(*layout)
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(name)
    plt.grid(False)
    plt.axis('off')


def show_range_image_channels(range_image_tensor, layout_index_start = 1):
    """Shows range image.

    Args:
        range_image_tensor: the range image data from a given lidar
            converted to a float tf.Tensor of the correct shape.
        layout_index_start: layout offset
    """
    lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
    range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                    tf.ones_like(range_image_tensor) * 1e10)
    range_image_range = range_image_tensor[...,0] 
    range_image_intensity = range_image_tensor[...,1]
    range_image_elongation = range_image_tensor[...,2]

    def adjust_im(im, p=0.5):
        # from scipy.signal import medfilt2d
        im = np.power(im, p)
        return im

    plot_range_image_helper(adjust_im(range_image_range.numpy(), 1), 'range',
                    [3, 1, layout_index_start], vmax=75, cmap='gray')
    plot_range_image_helper(adjust_im(range_image_intensity.numpy()), 'intensity',
                    [3, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
    plot_range_image_helper(adjust_im(range_image_elongation.numpy(), 0.25), 'elongation',
                    [3, 1, layout_index_start + 2], vmax=1.5, cmap='gray')


def samples_on_line(xyz1: np.ndarray, xyz2: np.ndarray, num: int=100) -> np.ndarray:
    assert xyz1.shape == (3,), "param `xyz1` must be a single 3D point"
    assert xyz2.shape == (3,), "param `xyz2` must be a single 3D point"
    xyz1, xyz2 = xyz1[None, :], xyz2[None, :]
    t = np.linspace(0, 1, num)[None, :]
    return np.einsum('ij,ik', t, xyz1) + np.einsum('ij,ik', 1 - t, xyz2)


def unit_square_3d(z: float) -> np.ndarray:
    return np.array([[-1., -1., z],
                     [+1., -1., z],
                     [+1., +1., z],
                     [-1., +1., z]], dtype=np.float32)

def _unit_box_3D():
    p, q = unit_square_3d(z=-1.), unit_square_3d(z=1.)
    return np.vstack([samples_on_line(a, b) for a, b in [(p[0], p[1]),
                                                         (p[1], p[2]),
                                                         (p[2], p[3]),
                                                         (p[3], p[0]),
                                                         (q[0], q[1]),
                                                         (q[1], q[2]),
                                                         (q[2], q[3]),
                                                         (q[3], q[0]),
                                                         (p[1], q[1]),
                                                         (p[2], q[2]),
                                                         (p[3], q[0]),
                                                         (p[0], q[3])]])
UNIT_BOX_POINTS_3D = _unit_box_3D()


def make_box_points(b) -> np.ndarray:
    a = b.heading
    R = np.array([[np.cos(a),  -np.sin(a),  0.],
                  [np.sin(a),   np.cos(a),  0.],
                  [        0,           0,  1.]])
    scale = np.r_[b.length, b.width, b.height] * .5
    offset = np.r_[b.center_x, b.center_y, b.center_z]
    p = UNIT_BOX_POINTS_3D * scale
    return ((R @ p.T).T + offset).astype(np.float32)


def export_data(frame_info: FrameInfo,
                out_dir: Path,
                filename_prefix: str,
                write_visualization_files: bool=False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # return info image tensor
    top_lidar_first_return_raw = frame_info.range_images[open_dataset.LaserName.TOP][0]
    top_lidar_first_returns = tf.convert_to_tensor(top_lidar_first_return_raw.data)
    top_lidar_first_returns = tf.reshape(top_lidar_first_returns,
                                top_lidar_first_return_raw.shape.dims)

    # indicator arrays for point labels
    cartesian = frame_utils.convert_range_image_to_cartesian(
        frame_info.src_frame,
        frame_info.range_images,
        frame_info.range_image_top_pose,
        ri_index=0)
    ri_shape = cartesian[open_dataset.LaserName.TOP].shape
    cartesian_flat = tf.reshape(cartesian[open_dataset.LaserName.TOP], (-1, 3))
    
    def box_info_array(b) -> np.array:
        return np.array(
            [b.center_x, b.center_y, b.center_z, b.length, b.width, b.height, b.heading],
            dtype=np.float32)

    label_boxes = np.vstack([box_info_array(a.box)
                                for a in frame_info.src_frame.laser_labels if a.HasField('box')])
    box_1hot = box_utils.is_within_box_3d(cartesian_flat, label_boxes).numpy()
    box_indices = (box_1hot * np.arange(len(label_boxes))).max(axis=-1)
    nobox = (box_1hot.any(axis=-1) == False)
    box_indices = np.where(nobox, np.full_like(box_indices, -1), box_indices)
    box_indices_reshaped = box_indices.reshape(*ri_shape[:2])

    # Save lidar return data
    out_data = np.dstack([top_lidar_first_returns.numpy(), box_indices_reshaped])
    np.save(out_dir / f'{filename_prefix}.npy', out_data, allow_pickle=False)

    if not write_visualization_files:
        return

    # Save a visualization of the range image
    fig = plt.figure(figsize=(64, 20))
    frame_info.src_frame.lasers.sort(key=lambda laser: laser.name)
    show_range_image_channels(top_lidar_first_returns, 1)
    plt.tight_layout()
    plt.savefig(Path(out_dir / f'{filename_prefix}.png'))
    plt.close(fig)

    # Convert the range image to a point cloud and save a .PLY
    TOP_LIDAR = 0
    points, _cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame_info.src_frame,
        frame_info.range_images,
        frame_info.camera_projections,
        frame_info.range_image_top_pose,
        keep_polar_features=True,
        ri_index=0)
    save_ply(points[TOP_LIDAR], Path(out_dir / f'{filename_prefix}_r0.ply'), transform=frame_info.world_from_frame)
    points, _cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame_info.src_frame,
        frame_info.range_images,
        frame_info.camera_projections,
        frame_info.range_image_top_pose,
        keep_polar_features=True,
        ri_index=1)
    save_ply(points[TOP_LIDAR], Path(out_dir / f'{filename_prefix}_r1.ply'), transform=frame_info.world_from_frame)

    # Save PLYs of boxes
    for i, a in enumerate(frame_info.src_frame.laser_labels):
        if not a.HasField('box'): continue
        p = make_box_points(a.box)
        rie_p = np.hstack([np.zeros((len(p), 3), dtype=np.float32), p])
        save_ply(rie_p, Path(out_dir) / f'{filename_prefix}.box{i:03d}.ply', transform=frame_info.world_from_frame)

    # Save a visualization of box labels on the lidar intensity image
    fig = plt.figure(figsize=(64, 20))
    intensity_im = top_lidar_first_returns[..., 1].numpy()
    intensity_im = intensity_im ** 0.5
    intensity_im = np.dstack([intensity_im, intensity_im, intensity_im])
    intensity_im[:,:,0] = (box_indices_reshaped > 0) * 255  # highlight labels in red
    plot_range_image_helper(intensity_im, 'labels', cmap=None)
    plt.savefig(out_dir / f'boxes.{filename_prefix}.png')
    plt.close(fig)


def main():
    FILENAME = '/code/waymo-od/tutorial/training_segment-10206293520369375008_2796_800_2816_800_with_camera_labels.tfrecord'
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    print(len([d for d in dataset]), 'frames')
    for i, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frame_info = FrameInfo(frame)
        print('Frame', i, frame.context.name)
        export_data(frame_info,
                    out_dir=Path(f'/tmp/waymo_od_lidar/{frame.context.name}'),
                    filename_prefix=f'{i:06d}')

if __name__ == '__main__':
    main()
