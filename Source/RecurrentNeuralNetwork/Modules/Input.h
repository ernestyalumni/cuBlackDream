#ifndef RECURRENT_NEURAL_NETWORK_MODULES_INPUT_H
#define RECURRENT_NEURAL_NETWORK_MODULES_INPUT_H

#include "RecurrentNeuralNetwork/Parameters.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cuda_runtime.h> // cudaFree, cudaMalloc
#include <stdexcept>

namespace RecurrentNeuralNetwork
{
namespace Modules
{

template <typename T>
class Input
{
  public:

    using HandleUnsuccessfulCUDACall =
      Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

    Input(RecurrentNeuralNetwork::Parameters& parameters):
      x_{nullptr}
    {
      HandleUnsuccessfulCUDACall handle_malloc {
        "Failed to allocate device memory for input"};

      handle_malloc(
        cudaMalloc(&x_, parameters.get_input_tensor_size() * sizeof(T)));

      if (!handle_malloc.is_cuda_success())
      {
        throw std::runtime_error(handle_malloc.get_error_message());
      }
    }

    ~Input()
    {
      HandleUnsuccessfulCUDACall handle_free_space {
        "Failed to free device memory for input"};

      handle_free_space(cudaFree(x_));
    }

    void* x_;
};

} // namespace Modules
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MODULES_INPUT_H