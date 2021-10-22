import argparse
from pathlib import Path

import tensorflow.compat.v1 as tf
import numpy as np

tf.enable_eager_execution()

from waymo_open_dataset.utils import frame_utils as futils
from waymo_open_dataset import dataset_pb2 as open_dataset


def transform_points(matrix_4x4: np.ndarray, points_MxNx3: np.ndarray) -> np.ndarray:
    original_shape = points_MxNx3.shape
    R, t = matrix_4x4[:3, :3], matrix_4x4[:3, 3:]
    return (R @ points_MxNx3.reshape(-1, 3).T + t).T.reshape(original_shape)


def main():
    parser = argparse.ArgumentParser(description='Dump range image data from a Waymo-OpenDataset dataset record')
    parser.add_argument('dataset_filepath', type=Path)
    parser.add_argument('--outdir', type=Path, default=None,
                        help='Output folder; defaults to a dir with name constructed from the arg to --dataset-filepath}')
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = args.dataset_filepath.with_suffix('.dump')
    dataset = tf.data.TFRecordDataset(str(args.dataset_filepath), compression_type='')

    for i, data in enumerate(dataset):
        print(args.dataset_filepath, ': frame', i)
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        range_images, _camera_projections, range_image_top_pose = futils.parse_range_image_and_camera_projection(frame)
        laser_to_returns = futils.convert_range_image_to_cartesian(frame,
                                                                   range_images,
                                                                   range_image_top_pose,
                                                                   ri_index=0,
                                                                   keep_polar_features=True)        
        # 64, 2650, 6:(range, intensity, elongation, x, y, z)
        top_lidar_returns_info = np.array(laser_to_returns[open_dataset.LaserName.TOP])

        world_from_vehicle = np.array(frame.pose.transform).reshape((4, 4))
        top_lidar_returns_info[:, :, 3:] = transform_points(world_from_vehicle, top_lidar_returns_info[:, :, 3:])

        args.outdir.mkdir(parents=True, exist_ok=True)
        outfile = args.outdir / f'{i:04d}.riexyz.npy'
        print(outfile, top_lidar_returns_info.shape, top_lidar_returns_info.dtype)
        np.save(outfile, top_lidar_returns_info)

if __name__ == '__main__':
    main()
