#ifndef UTILITIES_DEVICE_MANAGEMENT_GET_CUDA_DEVICE_PROPERTIES_H
#define UTILITIES_DEVICE_MANAGEMENT_GET_CUDA_DEVICE_PROPERTIES_H

#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

namespace Utilities
{
namespace DeviceManagement
{

class GetCUDADeviceProperties
{
  public:

    struct AbridgedProperties
    {
      // major
      int major_compute_capability_;
      // minor
      int minor_compute_capabilitiy_;
      // multiProcessorCount on device.
      int multi_processor_count_;
      // sharedMemPerMultiprocessor
      std::size_t shared_memory_bytes_;
      // totalGlobalMemory
      std::size_t total_global_memory_bytes_;
      // warpSize
      int warp_size_in_threads_;
    };

    GetCUDADeviceProperties();

    int get_device_count() const
    {
      return device_count_;
    }

    void pretty_print_abridged_properties();

    std::vector<cudaDeviceProp> cuda_device_properties_;

    std::vector<AbridgedProperties> abridged_properties_;

  private:

    int device_count_;
};

} // namespace DeviceManagement
} // namespace Utilities

#endif // UTILITIES_DEVICE_MANAGEMENT_GET_CUDA_DEVICE_PROPERTIES_H
