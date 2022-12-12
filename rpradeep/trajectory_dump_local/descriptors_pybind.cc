#include "descriptors.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

// Pack a TrackSegmentShapeDescriptor into a byte array / buffer.
py::bytes pyPackTrackShapeDescriptor(const TrackSegmentShapeDescriptor &d) {
  return std::string_view(reinterpret_cast<const char *>(&d), sizeof(d));
}

std::vector<TrackSegmentMatchInfo> pyFindNearest(
    const py::array_t<uint8_t, py::array::c_style> descr_buf,
    const py::array_t<float32_t, py::array::c_style> query_descr,
    const int num_neighbors) {
  const py::buffer_info descr_buf_info = descr_buf.request();
  CHECK_EQ(descr_buf_info.ndim, 1) << "Expected 1D array for `descr_buf`.";
  CHECK_EQ(descr_buf_info.size % sizeof(TrackSegmentShapeDescriptor), 0)
      << "Unexpected size for `descr_buf`.";

  const auto buf_begin = static_cast<TrackSegmentShapeDescriptor *>(descr_buf_info.ptr);
  const auto buf_size = descr_buf_info.size / sizeof(TrackSegmentShapeDescriptor);

  LOG(INFO) << "Searching over " << buf_size << " descriptor records.";
  return findNearest(buf_begin,
                     buf_begin + buf_size,
                     static_cast<const float32_t *>(query_descr.request().ptr),
                     num_neighbors);
}

PYBIND11_MODULE(descriptors, m) {
  m.doc() = "Module for processing track shape descriptions.";

  m.def("descr_array_size", [] { return TrackSegmentShapeDescriptor::kDescrArraySize; });

  using DescrArrayType = TrackSegmentShapeDescriptor::DescrArrayType;
  py::class_<TrackSegmentShapeDescriptor>(m, "TrackSegmentShapeDescriptor")
      .def(py::init<const DescrArrayType &, uint64_t, uint32_t, uint32_t>())
      .def_readonly("descr", &TrackSegmentShapeDescriptor::descr)
      .def_readonly("scenario_id", &TrackSegmentShapeDescriptor::scenario_id)
      .def_readonly("track_id", &TrackSegmentShapeDescriptor::track_id)
      .def_readonly("seg_id", &TrackSegmentShapeDescriptor::seg_id)
      .def("raw_bytes", &pyPackTrackShapeDescriptor);

  m.def("pack_track_shape_descr",
        &pyPackTrackShapeDescriptor,
        "Serialize a track shape description into bytes.");

  py::class_<TrackSegmentMatchInfo>(m, "TrackSegmentMatchInfo")
      .def_readonly("dist", &TrackSegmentMatchInfo::dist)
      .def_readonly("index", &TrackSegmentMatchInfo::index)
      .def_readonly("scenario_id", &TrackSegmentMatchInfo::scenario_id)
      .def_readonly("track_id", &TrackSegmentMatchInfo::track_id)
      .def_readonly("seg_id", &TrackSegmentMatchInfo::seg_id);

  m.def("find_nearest",
        &pyFindNearest,
        "Perform a linear search over a flat array of records to find the best matching "
        "descriptors.");
}
