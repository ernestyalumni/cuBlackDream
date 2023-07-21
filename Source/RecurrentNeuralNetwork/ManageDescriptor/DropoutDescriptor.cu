#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "DropoutDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cudnn.h>
#include <cuda_runtime.h> // cudaFree, cudaMallocManaged

using DeepNeuralNetwork::CuDNNLibraryHandle;
using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;
using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

DropoutDescriptor::DropoutDescriptor():
  descriptor_{},
  states_size_{0},
  states_{nullptr},
  is_states_size_known_{false}
{
  HandleUnsuccessfulCuDNNCall create_descriptor {
    "Failed to create dropout descriptor"};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateDropoutDescriptor
  // 3.2.9. cudnnCreateDropoutDescriptor(). This function creates a generic
  // dropout descriptor object by allocating memory needed to hold its opaque
  // structure.
  create_descriptor(cudnnCreateDropoutDescriptor(&descriptor_));
}

HandleUnsuccessfulCuDNNCall DropoutDescriptor::get_states_size_for_forward(
  CuDNNLibraryHandle& handle)
{
  HandleUnsuccessfulCuDNNCall query_states_size {
    "Failed to get states size used by cudnnDropoutForward()"};

  query_states_size(cudnnDropoutGetStatesSize(handle.handle_, &states_size_));

  is_states_size_known_ = true;

  HandleUnsuccessfulCUDACall handle_malloc {
    "Failed to allocate device memory for random number generator states"};

  handle_malloc(cudaMallocManaged(&states_, states_size_));

  return query_states_size;
}

DropoutDescriptor::~DropoutDescriptor()
{
  HandleUnsuccessfulCuDNNCall destroy_descriptor {
    "Failed to destroy dropout descriptor"};

  // https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroyDropoutDescriptor
  destroy_descriptor(cudnnDestroyDropoutDescriptor(descriptor_));

  HandleUnsuccessfulCUDACall handle_free_memory {
    "Failed to free device memory for random number generator states"};

  handle_free_memory(cudaFree(states_));
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork