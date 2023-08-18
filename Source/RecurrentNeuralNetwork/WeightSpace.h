#ifndef RECURRENT_NEURAL_NETWORK_WEIGHT_SPACE_H
#define RECURRENT_NEURAL_NETWORK_WEIGHT_SPACE_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cstddef>
#include <cuda_runtime.h>
#include <cudnn.h>

namespace RecurrentNeuralNetwork
{

class WeightSpace
{
  public:

    WeightSpace(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      RecurrentNeuralNetwork::ManageDescriptor::Descriptor& descriptor);

    WeightSpace(
      RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN&
        descriptors);

    ~WeightSpace();

    inline std::size_t get_weight_space_size() const
    {
      return weight_space_size_;
    }

    inline float get_weight_space_size_in_MiB() const
    {
      return static_cast<float>(weight_space_size_) / 1024.0 / 1024.0;
    }

    template <typename T>
    Utilities::ErrorHandling::HandleUnsuccessfulCUDACall copy_to_host(
      T* host_destination)
    {
      Utilities::ErrorHandling::HandleUnsuccessfulCUDACall handle_copy {
        "Failed to copy device weight space output to host"};

      handle_copy(cudaMemcpy(
        reinterpret_cast<void *>(&host_destination),
        d_weight_space_,
        weight_space_size_,
        cudaMemcpyDeviceToHost));

      return handle_copy;
    }

    void* weight_space_;

    //--------------------------------------------------------------------------
    /// \ref 8.2.22. cudnnRNNBackwardWeights_v8()
    /// \href https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNBackwardWeights_v8
    /// From 8.2.2.. cudnnRNNBackwardWeights_v8(),
    /// All gradient results (\partial \sigma_i/\partial w_j)^T \delta_{out}
    /// with respect to weights and biases are written to dweightSpace buffer.
    /// Size and organization of dweightSpace buffer is the same as weightSpace
    /// buffer that holds RNN weights and biases.
    /// Output. Address of weight space buffer in GPU memory.
    /// Currently, cudnnRNNBackwardWeights_v8() supports CUDNN_WGRAD_MODE_ADD
    /// mode only so dweightSpace buffer should be zeroed by user before
    /// invoking routine for first time.
    //--------------------------------------------------------------------------
    void* d_weight_space_;

  private:

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNWeightSpaceSize
    /// \param [in] handle - current cuDNN context handle.
    /// \parma [in] rnnDesc - previously initialized RNN descriptor.
    //--------------------------------------------------------------------------
    Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall get_weight_space_size(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      RecurrentNeuralNetwork::ManageDescriptor::Descriptor& descriptor);

    std::size_t weight_space_size_;
};

} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_WEIGHT_SPACE_H