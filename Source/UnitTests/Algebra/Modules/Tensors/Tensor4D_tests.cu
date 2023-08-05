#include "Algebra/Modules/Tensors/Tensor4D.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <vector>

using Algebra::Modules::Tensors::HostTensor4D;
using Algebra::Modules::Tensors::Tensor4D;
using std::size_t;
using std::vector;

namespace GoogleUnitTests
{
namespace Algebra
{
namespace Modules
{
namespace Tensors
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostTensor4DTests, Constructs)
{
  HostTensor4D ht {1, 2, 3, 10};

  EXPECT_EQ(ht.M_, 1);
  EXPECT_EQ(ht.N1_, 2);
  EXPECT_EQ(ht.N2_, 3);
  EXPECT_EQ(ht.N3_, 10);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostTensor4DTests, CopiesValuesWithValues)
{
  HostTensor4D<float> ht {1, 1, 3, 3};

  const vector<float> values {
    1.0f,
    2.0f,
    3.0f,
    4.0f,
    5.0f,
    6.0f,
    7.0f,
    8.0f,
    9.0f};

  const auto result = ht.copy_values(values);

  for (size_t j2 {0}; j2 < 3; ++j2)
  {
    for (size_t j3 {0}; j3 < 3; ++j3)
    {
      EXPECT_FLOAT_EQ(
        ht.get(0, 0, j2, j3),
        static_cast<float>(j2 * 3 + j3 + 1));
    }
  }

  EXPECT_EQ(result - ht.begin(), 9);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Tensor4DTests, Constructible)
{
  Tensor4D t4d {1, 2, 3, 4};

  EXPECT_EQ(t4d.M_, 1);
  EXPECT_EQ(t4d.N1_, 2);
  EXPECT_EQ(t4d.N2_, 3);
  EXPECT_EQ(t4d.N3_, 4);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(Tensor4DTests, Destructible)
{
  {
    Tensor4D t4d {1, 2, 3, 4};
  }

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(HostTensor4DTests, CopiesToDevice)
{
  {
    HostTensor4D<float> ht {1, 1, 3, 3};

    const vector<float> values {
      1.0f,
      2.0f,
      3.0f,
      4.0f,
      5.0f,
      6.0f,
      7.0f,
      8.0f,
      9.0f};

    auto result = ht.copy_values(values);

    Tensor4D<float> t4d {1, 1, 3, 3};

    t4d.copy_host_input_to_device(ht);

    HostTensor4D<float> ht_check {1, 1, 3, 3};
    const vector<float> empty_values (9, 0.0f);
    result = ht_check.copy_values(empty_values);

    t4d.copy_device_to_host(ht_check);

    for (size_t j2 {0}; j2 < 3; ++j2)
    {
      for (size_t j3 {0}; j3 < 3; ++j3)
      {
        EXPECT_FLOAT_EQ(
          ht_check.get(0, 0, j2, j3),
          static_cast<float>(j2 * 3 + j3 + 1));
      }
    }
  }
  {
    HostTensor4D<float> ht {1, 1, 1, 10};
    for (size_t i {0}; i < 10; ++i)
    {
      ht.get(0, 0, 0, i) = static_cast<float>(i);
    }

    for (size_t i {0}; i < 10; ++i)
    {
      EXPECT_FLOAT_EQ(ht.get(0, 0, 0, i), static_cast<float>(i));
    }

    Tensor4D<float> t4d {1, 1, 1, 10};

    t4d.copy_host_input_to_device(ht);

    HostTensor4D<float> ht_check {1, 1, 1, 10};
    const vector<float> empty_values (10, 0.0f);
    auto result = ht_check.copy_values(empty_values);

    t4d.copy_device_to_host(ht_check);

    for (size_t i {0}; i < 10; ++i)
    {
      EXPECT_FLOAT_EQ(ht_check.get(0, 0, 0, i), static_cast<float>(i));
    }
  }
}

} // namespace Tensors
} // namespace Modules
} // namespace Algebra
} // namespace GoogleUnitTests