#include "FrontEnd/Tensors/CreateTensor.h"
#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <cudnn.h>
#include <gmock/gmock.h> // ::testing::HasSubstr

using FrontEnd::Tensors::create_tensor;
using std::array;

namespace GoogleUnitTests
{

namespace FrontEnd
{

namespace Tensors
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CreateTensorsTests, BuildsWithArrayValues)
{
  // (Number of samples in Batch N_B, sequence length T, number of features D)
  const array<int64_t, 3> tensor_dim {100, 28, 28};
  const array<int64_t, 3> tensor_strides {784, 28, 1};

  auto tensor = create_tensor(
    CUDNN_DATA_FLOAT,
    0,
    tensor_dim,
    tensor_strides,
    sizeof(float));

  EXPECT_THAT(tensor.describe(), ::testing::HasSubstr(
    "CUDNN_BACKEND_TENSOR_DESCRIPTOR : Datatype: CUDNN_DATA_FLOAT Id: 0"));

  EXPECT_THAT(tensor.describe(), ::testing::HasSubstr(
    "Alignment: 4 nDims 3 VectorCount: 1 vectorDimension -1 Dim [ 100,28,28 "));

  EXPECT_THAT(tensor.describe(), ::testing::HasSubstr(
    "Str [ 784,28,1 ] isVirtual: 0 isByValue: 0"));

  EXPECT_THAT(tensor.describe(), ::testing::HasSubstr(
    "reorder_type: CUDNN_TENSOR_REORDERING_NONE"));

  // IsVirtual tells whether it's an intermediate tensor of an op (operation)
  // graph.
  EXPECT_FALSE(tensor.isVirtualTensor());
}

} // namespace Tensors

} // namespace FrontEnd
} // namespace GoogleUnitTests