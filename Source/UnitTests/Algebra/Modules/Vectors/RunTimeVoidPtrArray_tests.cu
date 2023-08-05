#include "Algebra/Modules/Vectors/RunTimeVoidPtrArray.h"
#include "Algebra/Modules/Vectors/VoidPtrArray.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

using Algebra::Modules::Vectors::HostArray;
using Algebra::Modules::Vectors::RunTimeVoidPtrArray;
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
TEST(RunTimeVoidPtrArrayTests, DefaultConstructs)
{
  RunTimeVoidPtrArray array {};

  EXPECT_EQ(array.values_, nullptr);
  EXPECT_EQ(array.total_size_, 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(RunTimeVoidPtrArrayTests, Initializes)
{
  RunTimeVoidPtrArray array {};

  static constexpr std::size_t N {1048576};

  const auto result = array.initialize(N * sizeof(float));

  EXPECT_TRUE(result.is_cuda_success());
  EXPECT_NE(array.values_, nullptr);
  EXPECT_EQ(array.total_size_, N * sizeof(float));
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(RunTimeVoidPtrArrayTests, CopiesToDevice)
{
  static constexpr std::size_t N {1048576};

  HostArray<float> x {N};

  for (size_t i {0}; i < N; ++i)
  {
    x.values_[i] = static_cast<float>(i) + 0.42f;
  }

  ASSERT_EQ(x.values_[0], 0.42f);
  ASSERT_EQ(x.values_[1], 1.42f);

  RunTimeVoidPtrArray d_x {};
  const auto result = d_x.initialize(N * sizeof(float));
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
TEST(RunTimeVoidPtrArrayTests, CopiesToDeviceWithStdVector)
{
  static constexpr std::size_t N {1048576};

  RunTimeVoidPtrArray array {};
  const auto result = array.initialize(N * sizeof(float));

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

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(RunTimeVoidPtrArrayTests, CopiesToDeviceFailsWithoutInitialization)
{
  static constexpr std::size_t N {1048576};

  HostArray<float> x {N};

  for (size_t i {0}; i < N; ++i)
  {
    x.values_[i] = static_cast<float>(i) + 0.42f;
  }

  ASSERT_EQ(x.values_[0], 0.42f);
  ASSERT_EQ(x.values_[1], 1.42f);

  RunTimeVoidPtrArray d_x {};
  EXPECT_FALSE(d_x.copy_host_input_to_device(x));
}

} // namespace Vectors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests