#ifndef ALGEBRA_MODULES_TENSORS_TENSOR4D_H
#define ALGEBRA_MODULES_TENSORS_TENSOR4D_H

#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h> // cudaFree, cudaMallocManaged
#include <vector>

namespace Algebra
{
namespace Modules
{

namespace Tensors
{

template <typename T = float>
class HostTensor4D
{
  public:

    HostTensor4D() = delete;

    HostTensor4D(
      const std::size_t M,
      const std::size_t N1,
      const std::size_t N2,
      const std::size_t N3
      ):
      values_(new T[M * N1 * N2 * N3]),
      M_{M},
      N1_{N1},
      N2_{N2},
      N3_{N3}
    {}

    ~HostTensor4D()
    {
      delete [] values_;
    }

    T& get(
      const std::size_t i,
      const std::size_t j1,
      const std::size_t j2,
      const std::size_t j3)
    {
      return values_[N3_ * N2_ * N1_ * i + (N3_ * N2_) * j1 + N3_ * j2 + j3];
    }

    T get(
      const std::size_t i,
      const std::size_t j1,
      const std::size_t j2,
      const std::size_t j3) const
    {
      return values_[N3_ * N2_ * N1_ * i + (N3_ * N2_) * j1 + N3_ * j2 + j3];
    }

    T* begin()
    {
      return values_;
    }

    T* end()
    {
      return values_ + total_number_of_elements();
    }

    const T* copy_values(const std::vector<T>& input_values)
    {
      return std::copy(
        input_values.begin(),
        input_values.end(),
        values_);
    }

    std::size_t total_number_of_elements() const
    {
      return M_ * N1_ * N2_ * N3_;
    }

    T* values_;

    const std::size_t M_;
    const std::size_t N1_;
    const std::size_t N2_;
    const std::size_t N3_;
};

template <typename T = float>
class Tensor4D
{
  public:

    using HandleUnsuccessfulCUDACall =
      Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

    Tensor4D() = delete;

    Tensor4D(
      const std::size_t M,
      const std::size_t N1,
      const std::size_t N2,
      const std::size_t N3
      ):
      values_{nullptr},
      M_{M},
      N1_{N1},
      N2_{N2},
      N3_{N3}
    {
      HandleUnsuccessfulCUDACall handle_malloc_values {
        "Failed to allocate device array for values"};

      //------------------------------------------------------------------------
      /// \ref https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gd228014f19cc0975ebe3e0dd2af6dd1b
      /// \brief Allocates bytes of managed memory on the device.
      //------------------------------------------------------------------------
      handle_malloc_values(
        cudaMallocManaged(
          reinterpret_cast<void**>(&values_),
          total_number_of_elements() * sizeof(T)));
    }

    ~Tensor4D()
    {
      HandleUnsuccessfulCUDACall handle_free_values {
        "Failed to free device array for values"};

      handle_free_values(cudaFree(values_));
    }

    T& get(
      const std::size_t i,
      const std::size_t j1,
      const std::size_t j2,
      const std::size_t j3)
    {
      return values_[N3_ * N2_ * N1_ * i + (N3_ * N2_) * j1 + N3_ * j2 + j3];
    }

    T get(
      const std::size_t i,
      const std::size_t j1,
      const std::size_t j2,
      const std::size_t j3) const
    {
      return values_[N3_ * N2_ * N1_ * i + (N3_ * N2_) * j1 + N3_ * j2 + j3];
    }

    //-------------------------------------------------------------------------=
    /// TODO: Understand why calling this doesn't work, resulting in compilation
    /// errors of incompatible type qualifiers, whereas calling the data member
    /// directly works.
    //-------------------------------------------------------------------------=
    T* begin()
    {
      return values_;
    }

    T* end()
    {
      return values_ + total_number_of_elements();
    }

    std::size_t total_number_of_elements() const
    {
      return M_ * N1_ * N2_ * N3_;
    }

    void copy_host_input_to_device(
      const HostTensor4D<T>& h_x)
    {
      HandleUnsuccessfulCUDACall handle_values {
        "Failed to copy values from host to device"};

      handle_values(cudaMemcpy(
        values_,
        h_x.values_,
        h_x.total_number_of_elements() * sizeof(T),
        cudaMemcpyHostToDevice));
    }

    void copy_device_to_host(HostTensor4D<T>& h_x)
    {
      HandleUnsuccessfulCUDACall handle_values {
        "Failed to copy values from device to host"};

      handle_values(cudaMemcpy(
        h_x.begin(),
        values_,
        total_number_of_elements() * sizeof(T),
        cudaMemcpyDeviceToHost));
    }

    const std::size_t M_;
    const std::size_t N1_;
    const std::size_t N2_;
    const std::size_t N3_;

    T* values_;
};

} // namespace Tensors

} // namespace Modules
} // namespace Algebra

#endif // ALGEBRA_MODULES_TENSORS_TENSOR4D_H