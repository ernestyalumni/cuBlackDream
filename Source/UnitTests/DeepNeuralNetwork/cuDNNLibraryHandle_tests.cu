#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "gtest/gtest.h"

using DeepNeuralNetwork::CuDNNLibraryHandle;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace DeviceManagement
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuDNNLibraryHandleTests, DefaultConstructs)
{
  CuDNNLibraryHandle handle {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(CuDNNLibraryHandleTests, Destructs)
{
  {
    CuDNNLibraryHandle handle {};
  }

  SUCCEED();
}

} // namespace DeviceManagement
} // namespace Utilities
} // namespace GoogleUnitTests