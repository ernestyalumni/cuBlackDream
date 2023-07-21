#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"
#include "WeightSpace.h"

#include <cstddef>
#include <cudnn.h>
#include <cuda_runtime.h> // cudaFree, cudaMallocManaged
#include <stdexcept>

using DeepNeuralNetwork::CuDNNLibraryHandle;
using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;
using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;
using RecurrentNeuralNetwork::ManageDescriptor::Descriptor;
using RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN;
using std::runtime_error;

namespace RecurrentNeuralNetwork
{

WeightSpace::WeightSpace(CuDNNLibraryHandle& handle, Descriptor& descriptor):
  weight_space_size_{0}
{
  const auto handle_get_weight_size = get_weight_space_size(handle, descriptor);

  if (!handle_get_weight_size.is_success())
  {
    throw runtime_error(handle_get_weight_size.get_error_message());
  }

  HandleUnsuccessfulCUDACall handle_malloc_1 {
    "Failed to allocate device memory for weight space"};

  handle_malloc_1(
    cudaMalloc(reinterpret_cast<void **>(&weight_space_), weight_space_size_));

  HandleUnsuccessfulCUDACall handle_malloc_2 {
    "Failed to allocate device memory for d(ifferential) weight space"};

  handle_malloc_2(
    cudaMalloc(
      reinterpret_cast<void **>(&d_weight_space_),
      weight_space_size_));

  if (!handle_malloc_1.is_cuda_success())
  {
    throw runtime_error(handle_malloc_1.get_error_message());
  }

  if (!handle_malloc_2.is_cuda_success())
  {
    throw runtime_error(handle_malloc_2.get_error_message());
  }
}

WeightSpace::WeightSpace(LibraryHandleDropoutRNN& descriptors):
  WeightSpace{descriptors.handle_, descriptors.descriptor_}
{}

WeightSpace::~WeightSpace()
{
  HandleUnsuccessfulCUDACall handle_free_space_1 {
    "Failed to free device memory for weight space"};

  handle_free_space_1(cudaFree(weight_space_));

  HandleUnsuccessfulCUDACall handle_free_space_2 {
    "Failed to free device memory for d(ifferential) weight space"};

  handle_free_space_2(cudaFree(d_weight_space_));
}

HandleUnsuccessfulCuDNNCall WeightSpace::get_weight_space_size(
  CuDNNLibraryHandle& handle,
  Descriptor& descriptor)
{
  HandleUnsuccessfulCuDNNCall handle_get_weight_size {
    "Failed to get weight size"};

  handle_get_weight_size(
    cudnnGetRNNWeightSpaceSize(
      handle.handle_,
      descriptor.descriptor_,
      &weight_space_size_));

  return handle_get_weight_size;
}

} // namespace RecurrentNeuralNetwork