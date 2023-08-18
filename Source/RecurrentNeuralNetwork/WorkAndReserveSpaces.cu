#include "WorkAndReserveSpaces.h"

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/DataDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/InputDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cuda_runtime.h> // cudaFree
#include <cudnn.h>
#include <stdexcept>

using DeepNeuralNetwork::CuDNNLibraryHandle;
using RecurrentNeuralNetwork::ManageDescriptor::DataDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::Descriptor;
using RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN;
using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;
using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;
using std::runtime_error;

namespace RecurrentNeuralNetwork
{

WorkAndReserveSpaces::WorkAndReserveSpaces(
  CuDNNLibraryHandle& handle,
  Descriptor& descriptor,
  DataDescriptor& data_descriptor,
  cudnnForwardMode_t forward_mode
  ):
  work_space_{nullptr},
  reserve_space_{nullptr},
  work_space_size_{},
  reserve_space_size_{},
  forward_mode_{forward_mode}
{
  HandleUnsuccessfulCuDNNCall handle_get_temp_sizes {get_temp_space_sizes(
    handle,
    descriptor,
    data_descriptor)};

  if (!handle_get_temp_sizes.is_success())
  {
    throw runtime_error(handle_get_temp_sizes.get_error_message());
  }

  HandleUnsuccessfulCUDACall handle_malloc_1 {
    "Failed to allocate device memory for work space"};

  handle_malloc_1(
    cudaMalloc(reinterpret_cast<void **>(&work_space_), work_space_size_));

  HandleUnsuccessfulCUDACall handle_malloc_2 {
    "Failed to allocate device memory for reserve space"};

  handle_malloc_2(
    cudaMalloc(
      reinterpret_cast<void **>(&reserve_space_),
      reserve_space_size_));

  if (!handle_malloc_1.is_cuda_success())
  {
    throw runtime_error(handle_malloc_1.get_error_message());
  }

  if (!handle_malloc_2.is_cuda_success())
  {
    throw runtime_error(handle_malloc_2.get_error_message());
  }
}

WorkAndReserveSpaces::WorkAndReserveSpaces(
  LibraryHandleDropoutRNN& descriptors,
  DataDescriptor& data_descriptor,
  cudnnForwardMode_t forward_mode
  ):
  WorkAndReserveSpaces{
    descriptors.handle_,
    descriptors.descriptor_,
    data_descriptor,
    forward_mode}
{}

WorkAndReserveSpaces::WorkAndReserveSpaces(
  LibraryHandleDropoutRNN& descriptors,
  InputDescriptor& x_data_descriptor,
  cudnnForwardMode_t forward_mode
  ):
  WorkAndReserveSpaces{
    descriptors.handle_,
    descriptors.descriptor_,
    x_data_descriptor.x_data_descriptor_,
    forward_mode}
{}

WorkAndReserveSpaces::~WorkAndReserveSpaces()
{
  HandleUnsuccessfulCUDACall handle_free_space_1 {
    "Failed to free device memory for work space"};

  handle_free_space_1(cudaFree(work_space_));

  HandleUnsuccessfulCUDACall handle_free_space_2 {
    "Failed to free device memory for reserve space"};

  handle_free_space_2(cudaFree(reserve_space_));
}

HandleUnsuccessfulCuDNNCall WorkAndReserveSpaces::get_temp_space_sizes(
  CuDNNLibraryHandle& handle,
  Descriptor& descriptor,
  DataDescriptor& data_descriptor)
{
  HandleUnsuccessfulCuDNNCall handle_get_temp_sizes {
    "Failed to get temp sizes for work and reserve spaces"};

  handle_get_temp_sizes(
    cudnnGetRNNTempSpaceSizes(
      handle.handle_,
      descriptor.descriptor_,
      forward_mode_,
      data_descriptor.descriptor_,
      &work_space_size_,
      &reserve_space_size_));

  return handle_get_temp_sizes;
}

} // namespace RecurrentNeuralNetwork
