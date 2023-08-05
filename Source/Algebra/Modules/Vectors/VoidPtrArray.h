#ifndef ALGEBRA_MODULES_VECTORS_VOID_PTR_ARRAY_H
#define ALGEBRA_MODULES_VECTORS_VOID_PTR_ARRAY_H

#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cstddef>
#include <cuda_runtime.h> // cudaFree, cudaMalloc, cudaMemcpyAsync
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace Algebra
{
namespace Modules
{
namespace Vectors
{

template <typename T>
struct HostArray
{
  T* values_;
  const std::size_t number_of_elements_;

  HostArray(const std::size_t input_size):
    values_{new T[input_size]},
    number_of_elements_{input_size}
  {}

  ~HostArray()
  {
    delete [] values_;
  }
};

template <typename T>
struct VoidPtrArray
{
  using HandleUnsuccessfulCUDACall =
    Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

  VoidPtrArray(const std::size_t input_size = 50000):
    values_{nullptr},
    number_of_elements_{input_size},
    type_identity_{}
  {
    HandleUnsuccessfulCUDACall handle_malloc {
      "Failed to allocate device array"};
    handle_malloc(
      cudaMalloc(reinterpret_cast<void**>(&values_), input_size * sizeof(T)));

    if (!handle_malloc.is_cuda_success())
    {
      throw std::runtime_error(handle_malloc.get_error_message());
    }
  }

  ~VoidPtrArray()
  {
    HandleUnsuccessfulCUDACall handle_free {"Failed to free device array"};
    handle_free(cudaFree(values_));
  }

  bool copy_host_input_to_device(const HostArray<T>& h_a)
  {
    HandleUnsuccessfulCUDACall handle_values {
      "Failed to copy values from host to device"};

    handle_values(cudaMemcpy(
      values_,
      h_a.values_,
      h_a.number_of_elements_ * sizeof(T),
      cudaMemcpyHostToDevice));

    return handle_values.is_cuda_success();
  }

  bool copy_host_input_to_device(const std::vector<T>& h_a)
  {
    HandleUnsuccessfulCUDACall handle_values {
      "Failed to copy values from host to device"};

    handle_values(cudaMemcpy(
      values_,
      h_a.data(),
      h_a.size() * sizeof(T),
      cudaMemcpyHostToDevice));

    return handle_values.is_cuda_success();
  }

  //------------------------------------------------------------------------------
  /// \details EY: 20230812 By C++20, the data pointer in a std::vector is going
  /// to be constexpr so we cannot mutate it. Use our defined container instead.
  //------------------------------------------------------------------------------
  bool copy_device_output_to_host(HostArray<T>& h_a)
  {
    HandleUnsuccessfulCUDACall handle_values {
      "Failed to copy values from device to host"};

    handle_values(cudaMemcpy(
      h_a.values_,
      values_,
      number_of_elements_ * sizeof(T),
      cudaMemcpyDeviceToHost));

    return handle_values.is_cuda_success();
  }

  std::type_identity<T> type_identity_;
  void* values_;
  const std::size_t number_of_elements_;
};

} // namespace Vectors
} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_VECTORS_VOID_PTR_ARRAY_H