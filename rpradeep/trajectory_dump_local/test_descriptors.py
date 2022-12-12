import unittest
import tempfile
import pathlib
import os

import glog
import numpy as np

import descriptors


class TestFindNearest(unittest.TestCase):

    def setUp(self):
        self._zero_descr = np.zeros(descriptors.descr_array_size(), dtype=np.float32)

        fd, self._tmpfile_path_str = tempfile.mkstemp(prefix="test_descriptors-unnitest-")
        with os.fdopen(fd, "wb") as f:
            NITEMS = 1024 * 1024
            for i in range(NITEMS):
                x = (i // 4096) + ((i % 4096) * (NITEMS // 4096))  # perfect-4096-shuffle
                v = [x] + ([0] * (descriptors.descr_array_size() - 1))
                f.write(descriptors.TrackSegmentShapeDescriptor(v, i, i, i).raw_bytes())

    def test_find_nearest(self):
        mmap = np.memmap(self._tmpfile_path_str, mode='r')
        nearest = descriptors.find_nearest(mmap, self._zero_descr, 4)
        self.assertEqual(nearest[0].dist, 0.)
        self.assertEqual(nearest[1].dist, 1.)
        self.assertEqual(nearest[2].dist, 2.)
        self.assertEqual(nearest[3].dist, 3.)

    def tearDown(self):
        pathlib.Path(self._tmpfile_path_str).unlink()


if __name__ == '__main__':
    unittest.main()
