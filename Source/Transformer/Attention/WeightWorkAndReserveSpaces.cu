#include "WeightWorkAndReserveSpaces.h"

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "Transformer/ManageDescriptor/AttentionDescriptor.h"
#include "Transformer/ManageDescriptor/LibraryHandleDropoutsAttention.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cuda_runtime.h> // cudaFree, cudaMallocManaged
#include <cudnn.h>
#include <stdexcept>

using DeepNeuralNetwork::CuDNNLibraryHandle;
using Transformer::ManageDescriptor::AttentionDescriptor;
using Transformer::ManageDescriptor::LibraryHandleDropoutsAttention;
using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;
using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;
using std::runtime_error;

namespace Transformer
{
namespace Attention
{

WeightWorkAndReserveSpaces::WeightWorkAndReserveSpaces(
  CuDNNLibraryHandle& handle,
  AttentionDescriptor& descriptor
  ):
  weight_space_{nullptr},
  work_space_{nullptr},
  reserve_space_{nullptr},
  weight_space_size_{0},
  work_space_size_{0},
  reserve_space_size_{0}
{
  const auto handle_get_buffer_sizes = get_buffer_sizes(handle, descriptor);

  if (!handle_get_buffer_sizes.is_success())
  {
    throw runtime_error(handle_get_buffer_sizes.get_error_message());
  }

  HandleUnsuccessfulCUDACall handle_malloc_1 {
    "Failed to allocate device memory for weight space"};

  handle_malloc_1(cudaMalloc(&weight_space_, weight_space_size_));

  if (!handle_malloc_1.is_cuda_success())
  {
    throw runtime_error(handle_malloc_1.get_error_message());
  }

  HandleUnsuccessfulCUDACall handle_malloc_2 {
    "Failed to allocate device memory for work space"};

  handle_malloc_2(cudaMalloc(&work_space_, work_space_size_));

  if (!handle_malloc_2.is_cuda_success())
  {
    throw runtime_error(handle_malloc_2.get_error_message());
  }

  HandleUnsuccessfulCUDACall handle_malloc_3 {
    "Failed to allocate device memory for reserve space"};

  handle_malloc_3(cudaMalloc(&reserve_space_, reserve_space_size_));

  if (!handle_malloc_3.is_cuda_success())
  {
    throw runtime_error(handle_malloc_3.get_error_message());
  }
}

WeightWorkAndReserveSpaces::WeightWorkAndReserveSpaces(
  LibraryHandleDropoutsAttention& descriptors
  ):
  WeightWorkAndReserveSpaces{descriptors.handle_, descriptors.descriptor_}
{}

WeightWorkAndReserveSpaces::~WeightWorkAndReserveSpaces()
{
  HandleUnsuccessfulCUDACall handle_free_space_1 {
    "Failed to free device memory for weight space"};

  handle_free_space_1(cudaFree(weight_space_));

  HandleUnsuccessfulCUDACall handle_free_space_2 {
    "Failed to free device memory for work space"};

  handle_free_space_2(cudaFree(work_space_));

  HandleUnsuccessfulCUDACall handle_free_space_3 {
    "Failed to free device memory for reserve space"};

  handle_free_space_3(cudaFree(reserve_space_));
}

HandleUnsuccessfulCuDNNCall WeightWorkAndReserveSpaces::get_buffer_sizes(
  CuDNNLibraryHandle& handle,
  AttentionDescriptor& descriptor)
{
  HandleUnsuccessfulCuDNNCall handle_get_buffer_sizes {
    "Failed to get buffer sizes"};

  handle_get_buffer_sizes(
    cudnnGetMultiHeadAttnBuffers(
      handle.handle_,
      descriptor.descriptor_,
      &weight_space_size_,
      &work_space_size_,
      &reserve_space_size_));

  return handle_get_buffer_sizes;
}

} // namespace Attention
} // namespace Transformers
