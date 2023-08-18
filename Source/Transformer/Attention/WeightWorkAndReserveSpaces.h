#ifndef TRANSFORMER_ATTENTION_WEIGHT_WORK_AND_RESERVE_SPACES_H
#define TRANSFORMER_ATTENTION_WEIGHT_WORK_AND_RESERVE_SPACES_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "Transformer/ManageDescriptor/AttentionDescriptor.h"
#include "Transformer/ManageDescriptor/LibraryHandleDropoutsAttention.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <cudnn.h>

namespace Transformer
{
namespace Attention
{

class WeightWorkAndReserveSpaces
{
  public:

    using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    WeightWorkAndReserveSpaces(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      Transformer::ManageDescriptor::AttentionDescriptor& descriptor);

    WeightWorkAndReserveSpaces(
      Transformer::ManageDescriptor::LibraryHandleDropoutsAttention&
        descriptors);

    ~WeightWorkAndReserveSpaces();

    inline std::size_t get_weight_space_size() const
    {
      return weight_space_size_;
    }

    inline std::size_t get_work_space_size() const
    {
      return work_space_size_;
    }

    inline std::size_t get_reserve_space_size() const
    {
      return reserve_space_size_;
    }

    template <typename T>
    Utilities::ErrorHandling::HandleUnsuccessfulCUDACall copy_weight_to_host(
      T* host_destination)
    {
      Utilities::ErrorHandling::HandleUnsuccessfulCUDACall handle_copy {
        "Failed to copy device weight space output to host"};

      handle_copy(cudaMemcpy(
        reinterpret_cast<void *>(&host_destination),
        weight_space_,
        weight_space_size_,
        cudaMemcpyDeviceToHost));

      return handle_copy;
    }

  private:

    //--------------------------------------------------------------------------
    /// \ref 7.2.15. cudnnGetMultiHeadAttnBuffers()
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetMultiHeadAttnBuffers
    /// User must allocate weight, work, and reserve space buffer sizes in GPU
    /// memory using cudaMalloc() with reported buffer sizes.
    //--------------------------------------------------------------------------
    void* weight_space_;
    void* work_space_;
    void* reserve_space_;

    HandleUnsuccessfulCuDNNCall get_buffer_sizes(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      Transformer::ManageDescriptor::AttentionDescriptor& descriptor);

    //--------------------------------------------------------------------------
    /// Minimum buffer size required to store all multi-head attention trainable
    /// parameters.
    //--------------------------------------------------------------------------
    std::size_t weight_space_size_;

    //--------------------------------------------------------------------------
    /// Minimum buffer size required to hold all temporary surfaces used by
    /// forward and gradient multi-head attention API calls.
    //--------------------------------------------------------------------------
    std::size_t work_space_size_;
    std::size_t reserve_space_size_;
};

} // namespace Attention
} // namespace Transformers

#endif // TRANSFORMER_ATTENTION_WEIGHT_WORK_AND_RESERVE_SPACES_H