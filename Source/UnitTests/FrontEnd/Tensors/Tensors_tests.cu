#include "cudnn_frontend.h"
#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <cudnn.h>

using std::array;

namespace GoogleUnitTests
{

namespace FrontEnd
{

namespace Tensors
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(TensorsTests, BuildsWithArrayValues)
{
  // (Number of samples in Batch N_B, sequence length T, number of features D)
  const array<int64_t, 3> tensor_dim {100, 28, 28};
  const array<int64_t, 3> tensor_strides {784, 28, 1};

  auto tensor = cudnn_frontend::TensorBuilder_v8()
    .setDim(tensor_dim.size(), tensor_dim.data())
    .setStrides(tensor_strides.size(), tensor_strides.data())
    .setId(0)
    .setAlignment(sizeof(float))
    .setDataType(CUDNN_DATA_FLOAT)
    .build();

  SUCCEED();
}

} // namespace Tensors

} // namespace FrontEnd
} // namespace GoogleUnitTests