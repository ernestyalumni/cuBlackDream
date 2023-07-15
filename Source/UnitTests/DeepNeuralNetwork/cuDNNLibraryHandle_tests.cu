#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "gtest/gtest.h"

#include <type_traits>

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

} // namespace DeviceManagement
} // namespace Utilities
} // namespace GoogleUnitTests