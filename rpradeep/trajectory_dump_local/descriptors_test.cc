#include "descriptors.h"

#include <gtest/gtest.h>

#include <boost/filesystem.hpp>
#include <cmath>

TEST(BestKHeap, Insertion) {
  detail::BestKHeap<int> heap(3);
  heap.maybeAdd(1);
  EXPECT_EQ(heap.elems().front(), 1);
  EXPECT_EQ(heap.elems().size(), 1);
  heap.maybeAdd(2);
  EXPECT_EQ(heap.elems().front(), 2);
  EXPECT_EQ(heap.elems().size(), 2);
  heap.maybeAdd(3);
  EXPECT_EQ(heap.elems().front(), 3);
  EXPECT_EQ(heap.elems().size(), 3);
  heap.maybeAdd(4);
  EXPECT_EQ(heap.elems().front(), 3);
  EXPECT_EQ(heap.elems().size(), 3);
  heap.maybeAdd(5);
  EXPECT_EQ(heap.elems().front(), 3);
  EXPECT_EQ(heap.elems().size(), 3);
  heap.maybeAdd(0);
  EXPECT_EQ(heap.elems().front(), 2);
  EXPECT_EQ(heap.elems().size(), 3);
  heap.maybeAdd(-1);
  EXPECT_EQ(heap.elems().front(), 1);
  EXPECT_EQ(heap.elems().size(), 3);
}

TEST(findNearest, AllPermuttionsSize8) {
  const TrackSegmentShapeDescriptor::DescrArrayType zero_descr{};

  // `descrs` contain descriptors that at distances 0-7 from the
  // all-zeros descriptor: `zero_descr`.
  std::vector<TrackSegmentShapeDescriptor> descrs;
  for (int i = 0; i < 8; ++i) {
    descrs.push_back({{float(i)}, uint64_t(i), uint32_t(i), uint32_t(i)});
  }

  for (int i = 0; i < tgamma(descrs.size()); ++i) {
    auto result = findNearest(descrs.data(), descrs.data() + descrs.size(), zero_descr, 4);
    std::sort(result.begin(), result.end());
    EXPECT_NEAR(result[0].dist, 0, 1e-6);
    EXPECT_NEAR(result[1].dist, 1, 1e-6);
    EXPECT_NEAR(result[2].dist, 2, 1e-6);
    EXPECT_NEAR(result[3].dist, 3, 1e-6);

    const auto& cmp = [](const auto& a, const auto& b) { return a.scenario_id < b.scenario_id; };
    std::next_permutation(descrs.begin(), descrs.end(), cmp);
  }
}

class findNearestPerformance : public ::testing::Test {
 protected:
  void SetUp() override {
    GTEST_SKIP() << "This test should be explicitly enabled by commenting out this line.";
    auto result = findNearest(descrs.data(), descrs.data() + descrs.size(), descrs.back().descr, 1);
  }

  const std::vector<TrackSegmentShapeDescriptor> descrs{32 * 1024 * 1024};
};

TEST_F(findNearestPerformance, Query64) {
  LOG(INFO) << "start";
  auto result = findNearest(descrs.data(), descrs.data() + descrs.size(), descrs.back().descr, 4);
  LOG(INFO) << "done";
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
