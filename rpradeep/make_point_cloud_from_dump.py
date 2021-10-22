import argparse
from collections import defaultdict
from pathlib import Path
from io import StringIO

import numpy as np


def ExistingFilePath(s: str) -> Path:
    p = Path(s)
    if not p.is_file():
        raise ValueError(f"Invalid input file path: {p}")
    return p


def main():
    parser = argparse.ArgumentParser(description='Group range data into voxels')
    parser.add_argument('dataset_dump_files', nargs='+', type=ExistingFilePath)
    args = parser.parse_args()

    points = []
    for range_info_filepath in sorted(Path(x) for x in args.dataset_dump_files):
        print(f'Loading {range_info_filepath} ...')
        # 64, 2650, 6:(range, intensity, elongation, x, y, z)
        range_info = np.load(range_info_filepath)[0:22, :, :]
        points.append(range_info.reshape(-1, 6)[:, 3:])
    points = np.vstack(points)

    print(f'Writing PLY with {len(points)} points...')
    with Path('pointcloud.ply').open('wb') as f:
        f.write(("ply\n" +
                 "format binary_little_endian 1.0\n" +
                f"element vertex {len(points)}\n" +
                 "property float x\n" +
                 "property float y\n" +
                 "property float z\n" +
                 "end_header\n").encode('ascii'))
        f.write(points.tobytes())
    print('Done!')

if __name__ == '__main__':
    main()
