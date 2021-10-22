import argparse
from collections import defaultdict
from pathlib import Path
from io import StringIO

import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Group range data into voxels')
    parser.add_argument('dataset_dump_files', nargs='+')
    args = parser.parse_args()

    voxels = defaultdict(list)
    BLOCK_SIZE = 10
    VOXEL_SIZE = 1

    for range_info_filepath in sorted(Path(x) for x in args.dataset_dump_files):
        print(f'Loading {range_info_filepath} ...')
        # 64, 2650, 6:(range, intensity, elongation, x, y, z)
        range_info = np.load(range_info_filepath)
        H, W, *_ = range_info.shape
        for b in range(0, W, BLOCK_SIZE):
            centroids = (range_info[:, b: b+BLOCK_SIZE, 3:].mean(axis=1) // VOXEL_SIZE).astype(int)
            for c, data in zip(centroids, range_info[:, b: b+BLOCK_SIZE, :]):
                voxels[tuple(c)].append(data)

    num_points = sum(len(scan) for scan_list in voxels.values() for scan in scan_list)
    print(f'Indexed {num_points} points')

    print('Writing PLY ...')
    with Path('voxels.ply').open('wb') as f:
        f.write(("ply\n" +
                 "format binary_little_endian 1.0\n" +
                f"element vertex {num_points}\n" +
                 "property float x\n" +
                 "property float y\n" +
                 "property float z\n" +
                 "property uchar red\n" +
                 "property uchar green\n" +
                 "property uchar blue\n"
                 "property uchar _pad\n"
                 "end_header\n").encode('ascii'))
        num_bytes = 0
        for voxel_id, scan_list in voxels.items():
            c = np.random.randint(0, 256, (1, 4)).astype(np.uint8)
            for scan in scan_list:
                color = c.repeat(len(scan), axis=0).view(np.float32)
                assert scan.dtype == np.float32
                data = np.hstack([scan[:, 3:], color])
                data_bytes = data.tobytes()
                assert len(data_bytes) == len(scan) * 4 * 4, f'{data.shape} {len(data_bytes)} != {len(scan) * 4 * 4}'
                f.write(data_bytes)
                num_bytes += len(data_bytes)

    print(f'{num_bytes} bytes of data written to PLY')
    print('Done!')

if __name__ == '__main__':
    main()
