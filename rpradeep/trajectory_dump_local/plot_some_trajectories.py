import argparse
import pathlib
import typing as T

from matplotlib import pyplot as plt
import numpy as np

import descriptors
from trajectory_util import generate_track_array_segments, compute_shape_descriptor


def plot_track_trajectory(txy: np.ndarray, mark_seg_indices: bool = False):
    t, x, y = txy.T
    plt.gca().set_aspect('equal', adjustable='datalim')
    if mark_seg_indices:
        for i, (xx, yy) in enumerate(zip(x, y)):
            if i % 10 != 0: continue
            plt.text(xx + .5, yy + .5, i, color='#ddd')

    plt.plot(x, y, '-', color='#0008', linewidth=0.5)
    plt.scatter(x, y, c=t, marker='.', cmap='OrRd')
    last_x, last_y = None, None
    for xx, yy in zip(x, y):
        if np.isnan(xx) or np.isnan(yy): continue
        last_x, last_y = xx, yy
    if last_x is not None and last_y is not None:
        plt.plot(last_x, last_y, 'o', markeredgecolor='red', markerfacecolor='#faf', markersize=6)
    plt.gca().axis('off')


def plot_track_segment(xy: np.ndarray, seg_index: T.Optional[int]):
    x, y = xy.T
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.plot(x, y, '-', color='steelblue')
    plt.plot(x, y, marker='o', markeredgecolor='r', markerfacecolor='w', markersize=8)
    plt.plot(x[-1], y[-1], marker='o', markeredgecolor='r', markerfacecolor='w', markersize=8 * 1.6)
    plt.plot(x[-1],
             y[-1],
             marker='o',
             markeredgecolor='r',
             markerfacecolor='#f008',
             markersize=8 / 1.6)
    if seg_index is not None:
        plt.text(1,
                 1,
                 seg_index,
                 fontsize=14,
                 color='#ccc',
                 horizontalalignment='right',
                 verticalalignment='top',
                 transform=plt.gca().transAxes)
    plt.gca().axis('off')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_npz_path", type=pathlib.Path, help="Output directory.")
    parser.add_argument("--plot-segs", type=bool, default=False)
    args = parser.parse_args()

    # "trajectories/scen:fb3d32567493900e.npz"
    npz = np.load(args.input_npz_path)
    fig = plt.Figure(figsize=(16, 16))
    for track in npz.files:
        txy = npz[track]
        plt.clf()
        plot_track_trajectory(txy)
        plt.savefig(f"/tmp/plots/{track}.png")

        if not args.plot_segs: continue
        last_descr = np.zeros(descriptors.descr_array_size(), np.float32)
        for i, seg in enumerate(generate_track_array_segments(txy[:, 1:])):
            if (np.isnan(seg).any()): continue
            descr = compute_shape_descriptor(seg[:, :2])
            if np.linalg.norm(descr - last_descr) < 0.4:
                continue
            last_descr = descr
            plt.clf()
            plot_track_segment(seg, i)
            plt.savefig(f"/tmp/plots/{track}--seg{i:03d}.png")


if __name__ == '__main__':
    main()
