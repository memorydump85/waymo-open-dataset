#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>

#define CHECK_EQ(val, ref, msg) \
  if ((val) != (ref)) throw std::runtime_error((msg));

namespace py = pybind11;
using float32_t = float;

// A decsription of the shape of a track segment
struct __attribute__((packed)) TrackSegmentShapeDescriptor {
  static constexpr std::size_t kDescrArraySize = 12;

  float32_t descr[kDescrArraySize];  // Vector description / embedding of track shape.
  uint64_t scenario_id;              // Scenario ID of the scenario this track belongs to.
  uint32_t track_id;                 // The track unique id.
  uint32_t seg_id;                   // The (overlapping) segment number of track.
};

static_assert(sizeof(size_t) == 8,  //
              "This module assumes 64 bit platform.");
static_assert(sizeof(TrackSegmentShapeDescriptor) % sizeof(size_t) == 0,
              "sizeof(TrackSegmentShapeDescriptor) must be exactly aligned "
              "with the machine word size.");

// Pack a TrackSegmentShapeDescriptor into a byte array / buffer.
py::bytes pyPackTrackShapeDescriptor(const py::array_t<float32_t, py::array::c_style> descr,
                                     const uint64_t scenario_id, const uint32_t track_id,
                                     const uint32_t seg_id) {
  const py::buffer_info descr_info = descr.request();
  CHECK_EQ(descr_info.ndim, 1, "Expected 1D array for `descr`.");
  CHECK_EQ(descr_info.size, TrackSegmentShapeDescriptor::kDescrArraySize,
           "Unexpected size for `descr`.");

  TrackSegmentShapeDescriptor output{
      {}, .scenario_id = scenario_id, .track_id = track_id, .seg_id = seg_id};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"
  std::copy_n(static_cast<float32_t*>(descr_info.ptr), TrackSegmentShapeDescriptor::kDescrArraySize,
              output.descr);
#pragma GCC diagnostic pop

  return std::string_view(reinterpret_cast<char*>(&output), sizeof(output));
}

//
// Binding definition
//

PYBIND11_MODULE(descriptors, m) {
  m.doc() = "Module for processing track shape descriptions.";

  m.def("pack_track_shape_descr", &pyPackTrackShapeDescriptor,
        "Serialize a track shape description into bytes.");
}
