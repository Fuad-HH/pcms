#include <wdmcpl/xgc_reverse_classification.h>
#include <catch2/catch_test_macros.hpp>
#include <sstream>

static constexpr auto test_data = R"(
17
0 1
13
0 2
37
1 11
734 735 736 737
1 12
871 872 873 874
1 13
1020 1021 1022 1023
1 22
-1
2 20
-1
2 1
5 6 7
)";

TEST_CASE("reverse classification") {
  std::stringstream ss{test_data};
  auto rc = wdmcpl::ReadReverseClassificationVertex(ss);
  {
    const auto* q = rc.Query({2,20});
    REQUIRE(q == nullptr);
  }
  {
    const auto* q = rc.Query({2,1});
    REQUIRE(q!=nullptr);
    REQUIRE(q->size() == 3);
    std::set<wdmcpl::LO> s{4, 5, 6};
    REQUIRE(s == *q);
  }
  auto vec = rc.Serialize();
  wdmcpl::ReverseClassificationVertex rc_deserialized;
  wdmcpl::ScalarArrayView<wdmcpl::LO, wdmcpl::HostMemorySpace> av{vec.data(),vec.size()};
  rc_deserialized.Deserialize(av);
  REQUIRE(rc_deserialized == rc);
}
