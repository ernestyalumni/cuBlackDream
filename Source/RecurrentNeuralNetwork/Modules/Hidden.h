#ifndef RECURRENT_NEURAL_NETWORK_MODULES_HIDDEN_H
#define RECURRENT_NEURAL_NETWORK_MODULES_HIDDEN_H

#include "RecurrentNeuralNetwork/Parameters.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cuda_runtime.h> // cudaFree, cudaMalloc
#include <stdexcept>

namespace RecurrentNeuralNetwork
{
namespace Modules
{

template <typename T>
class Hidden
{
  public:

    using HandleUnsuccessfulCUDACall =
      Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

    Hidden(const RecurrentNeuralNetwork::Parameters& parameters):
      h_{nullptr}
    {
      HandleUnsuccessfulCUDACall handle_malloc {
        "Failed to allocate device memory for hidden"};

      handle_malloc(
        cudaMalloc(&h_, parameters.get_hidden_tensor_size() * sizeof(T)));

      if (!handle_malloc.is_cuda_success())
      {
        throw std::runtime_error(handle_malloc.get_error_message());
      }
    }

    ~Hidden()
    {
      HandleUnsuccessfulCUDACall handle_free_space {
        "Failed to free device memory for hidden"};

      handle_free_space(cudaFree(h_));
    }

    void* h_;
};

} // namespace Modules
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MODULES_HIDDEN_H