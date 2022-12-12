#include "descriptors.h"

#include <cstdint>

#include "boost/iostreams/device/mapped_file.hpp"
#include "eigen3/Eigen/Core"

std::vector<TrackSegmentMatchInfo> findNearest(
    const TrackSegmentShapeDescriptor* descr_flat_collection_begin,
    const TrackSegmentShapeDescriptor* descr_flat_collection_end,
    const float32_t* query,
    const size_t num_neighbors) {
  CHECK_NOTNULL(descr_flat_collection_begin);
  CHECK_NOTNULL(descr_flat_collection_end);
  CHECK_NOTNULL(query);
  using DescrVectorType = Eigen::Matrix<float32_t, 1, TrackSegmentShapeDescriptor::kDescrArraySize>;
  detail::BestKHeap<TrackSegmentMatchInfo> neighbors(num_neighbors);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waddress-of-packed-member"
  Eigen::Map<const DescrVectorType> q(query);
  for (auto it = descr_flat_collection_begin; it != descr_flat_collection_end; ++it) {
    Eigen::Map<const DescrVectorType> candidate(it->descr.data());
    const auto dist = (q - candidate).norm();
    neighbors.maybeAdd(TrackSegmentMatchInfo{.dist = dist,
                                             .index = it - descr_flat_collection_begin,
                                             .scenario_id = it->scenario_id,
                                             .track_id = it->track_id,
                                             .seg_id = it->seg_id});
  }
#pragma GCC diagnostic pop

  auto result = neighbors.elems();
  std::sort(result.begin(), result.end());
  return result;
}
