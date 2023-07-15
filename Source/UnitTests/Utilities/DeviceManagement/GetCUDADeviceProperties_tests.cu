#include "Utilities/DeviceManagement/GetCUDADeviceProperties.h"
#include "gtest/gtest.h"

#include <type_traits>

using Utilities::DeviceManagement::GetCUDADeviceProperties;

namespace GoogleUnitTests
{
namespace Utilities
{
namespace DeviceManagement
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(GetCUDADevicePropertiesTests, DefaultConstructible)
{
  GetCUDADeviceProperties device_properties {};

  EXPECT_TRUE(device_properties.get_device_count() > 0);
  EXPECT_TRUE(device_properties.cuda_device_properties_.size() > 0);
  EXPECT_TRUE(device_properties.abridged_properties_.size() > 0);
}

} // namespace DeviceManagement
} // namespace Utilities
} // namespace GoogleUnitTests