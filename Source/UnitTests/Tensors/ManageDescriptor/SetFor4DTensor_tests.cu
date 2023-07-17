#include "Tensors/ManageDescriptor/SetFor4DTensor.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "gtest/gtest.h"

using Tensors::ManageDescriptor::SetFor4DTensor;
using Tensors::ManageDescriptor::TensorDescriptor;

namespace GoogleUnitTests
{
namespace Tensors
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetFor4DTensorTests, Constructs)
{
  SetFor4DTensor set4d {1, 1, 1, 10, CUDNN_DATA_FLOAT};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetFor4DTensorTests, SetDescriptorSetsForCudnnTensorDescriptorT)
{
  TensorDescriptor descriptor {};

  SetFor4DTensor set4d {1, 1, 1, 10, CUDNN_DATA_FLOAT};

  set4d.set_descriptor(descriptor.descriptor_);

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(SetFor4DTensorTests, SetDescriptorSetsForTensorDescriptor)
{
  TensorDescriptor descriptor {};

  SetFor4DTensor set4d {1, 1, 1, 10, CUDNN_DATA_FLOAT};

  set4d.set_descriptor(descriptor);

  SUCCEED();
}

} // namespace ManageDescriptor
} // namespace Tensors
} // namespace GoogleUnitTests