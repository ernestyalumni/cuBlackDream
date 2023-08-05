#include "Algebra/Modules/Vectors/VoidPtrArray.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

using Algebra::Modules::Vectors::HostArray;
using Algebra::Modules::Vectors::VoidPtrArray;
using std::size_t;
using std::vector;

namespace GoogleUnitTests
{
namespace Algebra
{
namespace Modules
{
namespace Vectors
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(VoidPtrArrayTests, Constructs)
{
  static constexpr std::size_t N {1048576};

  VoidPtrArray<float> array {N};
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(VoidPtrArrayTests, CopiesToDevice)
{
  static constexpr std::size_t N {1048576};

  HostArray<float> x {N};

  for (size_t i {0}; i < N; ++i)
  {
    x.values_[i] = static_cast<float>(i) + 0.42f;
  }

  ASSERT_EQ(x.values_[0], 0.42f);
  ASSERT_EQ(x.values_[1], 1.42f);

  VoidPtrArray<float> d_x {N};
  EXPECT_TRUE(d_x.copy_host_input_to_device(x));

  HostArray<float> x_out {N};

  for (size_t i {0}; i < N; ++i)
  {
    x_out.values_[i] = 69.0f;
  }

  EXPECT_TRUE(d_x.copy_device_output_to_host(x_out));

  for (size_t i {0}; i < N; ++i)
  {
    EXPECT_FLOAT_EQ(x_out.values_[i], static_cast<float>(i) + 0.42f);
  }
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(VoidPtrArrayTests, CopiesToDeviceWithStdVector)
{
  static constexpr std::size_t N {1048576};

  VoidPtrArray<float> array {N};

  vector<float> h_array (N, 0.0f);

  std::generate(
    h_array.begin(),
    h_array.end(),
    [x = 0.0f]() mutable
    {
      x += 1.0f;
      return x;
    });

  ASSERT_EQ(h_array.at(0), 1.0f);
  ASSERT_EQ(h_array.at(1), 2.0f);

  EXPECT_TRUE(array.copy_host_input_to_device(h_array));

  HostArray<float> x_out {N};

  for (size_t i {0}; i < N; ++i)
  {
    x_out.values_[i] = 69.0f;
  }

  EXPECT_TRUE(array.copy_device_output_to_host(x_out));

  for (size_t i {0}; i < N; ++i)
  {
    EXPECT_FLOAT_EQ(x_out.values_[i], static_cast<float>(i) + 1.0f);
  }
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests