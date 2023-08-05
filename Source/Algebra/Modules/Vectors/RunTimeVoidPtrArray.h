#ifndef ALGEBRA_MODULES_VECTORS_RUN_TIME_VOID_PTR_ARRAY_H
#define ALGEBRA_MODULES_VECTORS_RUN_TIME_VOID_PTR_ARRAY_H

#include "VoidPtrArray.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cstddef>
#include <cuda_runtime.h> // cudaFree, cudaMalloc, cudaMemcpyAsync
#include <vector>

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

//------------------------------------------------------------------------------
/// \details Because of the CUDNN API, in particular the cudnnGetRNNWeightParams
/// call, you won't know the type and size of objects before run-time (because
/// you'll call the function at run-time to get the weight parameters). Thus,
/// we made this struct.
//------------------------------------------------------------------------------
struct RunTimeVoidPtrArray
{
  using HandleUnsuccessfulCUDACall =
    Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

  RunTimeVoidPtrArray();

  ~RunTimeVoidPtrArray();

  //----------------------------------------------------------------------------
  /// \param [in] total_size - total number of bytes to initialize by. So you'll
  /// have to take the size of each element, i.e. the size of the type, and
  /// multiply by number of elements.
  //----------------------------------------------------------------------------
  HandleUnsuccessfulCUDACall initialize(const std::size_t total_size);

  template <typename T>
  bool copy_host_input_to_device(const HostArray<T>& h_a)
  {
    if (values_ == nullptr)
    {
      return false;
    }

    HandleUnsuccessfulCUDACall handle_values {
      "Failed to copy values from host to device"};

    handle_values(cudaMemcpy(
      values_,
      h_a.values_,
      h_a.number_of_elements_ * sizeof(T),
      cudaMemcpyHostToDevice));

    return handle_values.is_cuda_success();    
  }

  template <typename T>
  bool copy_host_input_to_device(const std::vector<T>& h_a)
  {
    if (values_ == nullptr)
    {
      return false;
    }

    HandleUnsuccessfulCUDACall handle_values {
      "Failed to copy values from host to device"};

    handle_values(cudaMemcpy(
      values_,
      h_a.data(),
      h_a.size() * sizeof(T),
      cudaMemcpyHostToDevice));

    return handle_values.is_cuda_success();    
  }

  template <typename T>
  bool copy_device_output_to_host(HostArray<T>& h_a)
  {
    if (values_ == nullptr)
    {
      return false;
    }

    HandleUnsuccessfulCUDACall handle_values {
      "Failed to copy values from device to host"};

    handle_values(cudaMemcpy(
      h_a.values_,
      values_,
      total_size_,
      cudaMemcpyDeviceToHost));

    return handle_values.is_cuda_success();    
  }

  void* values_;
  std::size_t total_size_;
};

} // namespace Vectors
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_VECTORS_RUN_TIME_VOID_PTR_ARRAY_H