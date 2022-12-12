#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "glog/logging.h"

using float32_t = float;

// A decsription of the shape of a track segment
struct __attribute__((packed)) TrackSegmentShapeDescriptor {
  static constexpr std::size_t kDescrArraySize = 12;
  using DescrArrayType = std::array<float32_t, kDescrArraySize>;

  DescrArrayType descr;  // Vector description / embedding of track shape.
  uint64_t scenario_id;  // Scenario ID of the scenario this track belongs to.
  uint32_t track_id;     // The track unique id.
  uint32_t seg_id;       // The (overlapping) segment number of track.
};

static_assert(sizeof(std::size_t) == 8,  //
              "This module assumes 64 bit platform.");
static_assert(sizeof(TrackSegmentShapeDescriptor) % sizeof(std::size_t) == 0,
              "sizeof(TrackSegmentShapeDescriptor) must be exactly aligned "
              "with the machine word size.");

// How close is a track-segment to a specific reference / query segment?
struct TrackSegmentMatchInfo {
  float32_t dist;
  int64_t index;
  uint64_t scenario_id;
  uint32_t track_id;
  uint32_t seg_id;

  bool operator<(const TrackSegmentMatchInfo& other) const { return this->dist < other.dist; }
};

// Performa a linear search through the contents of
// `descr_flat_collection_path` for `num_neighbors` nearest descriptors
// that match the descriptor portion of `query_descr_raw_bytes`.
std::vector<TrackSegmentMatchInfo> findNearest(
    const TrackSegmentShapeDescriptor* descr_flat_collection_begin,
    const TrackSegmentShapeDescriptor* descr_flat_collection_end,
    const float32_t* query,
    const size_t num_neighbors);

template <class SeqType>
std::vector<TrackSegmentMatchInfo> findNearest(
    const TrackSegmentShapeDescriptor* descr_flat_collection_begin,
    const TrackSegmentShapeDescriptor* descr_flat_collection_end,
    const SeqType& query,
    const size_t num_neighbors) {
  return findNearest(
      descr_flat_collection_begin, descr_flat_collection_end, query.data(), num_neighbors);
}

namespace detail {

// Use a binary heap to maintain the K best elements (as per
// T::operator <) from a stream of input elements.
// TOOD(rpradeep): would a std::partition based implemnentation be
// faster for smaller values of `max_size=K`?
template <class T>
class BestKHeap {
 public:
  BestKHeap(const size_t max_size) : max_size_(max_size) { elems_.reserve(max_size_ + 1); }

  // Add `t` into this heap if it is lesser than at least one existing
  // element.
  void maybeAdd(const T& t) {
    CHECK_LE(elems_.size(), max_size_);

    elems_.emplace_back(t);
    std::push_heap(elems_.begin(), elems_.end());
    if (elems_.size() > max_size_) {
      std::pop_heap(elems_.begin(), elems_.end());
      elems_.pop_back();
    }

    CHECK_LE(elems_.size(), max_size_);
  }

  // Get the contained sequence of elements. The resulting sequence will
  // always be sorted in ascending order.
  const auto& elems() const { return elems_; }

 private:
  const size_t max_size_;
  std::vector<T> elems_;
};

}  // namespace detail
