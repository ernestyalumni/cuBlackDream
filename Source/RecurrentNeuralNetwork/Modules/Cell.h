#ifndef RECURRENT_NEURAL_NETWORK_MODULES_CELL_H
#define RECURRENT_NEURAL_NETWORK_MODULES_CELL_H

#include "RecurrentNeuralNetwork/Parameters.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cuda_runtime.h> // cudaFree, cudaMalloc
#include <stdexcept>

namespace RecurrentNeuralNetwork
{
namespace Modules
{

template <typename T>
class Cell
{
  public:

    using HandleUnsuccessfulCUDACall =
      Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

    Cell(const RecurrentNeuralNetwork::Parameters& parameters):
      c_{nullptr}
    {
      HandleUnsuccessfulCUDACall handle_malloc {
        "Failed to allocate device memory for cell"};

      handle_malloc(
        cudaMalloc(&c_, parameters.get_cell_tensor_size() * sizeof(T)));

      if (!handle_malloc.is_cuda_success())
      {
        throw std::runtime_error(handle_malloc.get_error_message());
      }
    }

    ~Cell()
    {
      HandleUnsuccessfulCUDACall handle_free_space {
        "Failed to free device memory for cell"};

      handle_free_space(cudaFree(c_));
    }

    void* c_;
};

} // namespace Modules
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MODULES_CELL_H