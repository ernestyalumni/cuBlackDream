#include "Parameters.h"
#include "SequenceLengthArray.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCudaCall.h"

#include <cstddef>
#include <cstdint>
#include <cudnn.h>
#include <vector>

using Utilities::ErrorHandling::HandleUnsuccessfulCUDACall;

namespace RecurrentNeuralNetwork
{

HostSequenceLengthArray::HostSequenceLengthArray(const Parameters& parameters):
  sequence_length_array_{new int[parameters.batch_size_]},
  batch_size_{parameters.batch_size_},
  maximum_sequence_length_{parameters.maximum_sequence_length_}
{}

HostSequenceLengthArray::~HostSequenceLengthArray()
{
  delete [] sequence_length_array_;
}

void HostSequenceLengthArray::copy_values(
  const std::vector<int>& input_values)
{
  for (std::size_t i {0}; i < batch_size_; ++i)
  {
    const int input_value {input_values.at(i)};
    if (input_value > maximum_sequence_length_)
    {
      sequence_length_array_[i] = maximum_sequence_length_;
    }
    else
    {
      sequence_length_array_[i] = input_value;
    }
  }
}

void HostSequenceLengthArray::set_all_to_maximum_sequence_length()
{
  for (std::size_t i {0}; i < batch_size_; ++i)
  {
    sequence_length_array_[i] = maximum_sequence_length_;
  }
}

SequenceLengthArray::SequenceLengthArray(const Parameters& parameters):
  sequence_length_array_{nullptr},
  batch_size_{parameters.batch_size_},
  maximum_sequence_length_{parameters.maximum_sequence_length_}
{
  HandleUnsuccessfulCUDACall handle_malloc {
    "Failed to allocate device array for sequence length array"};

  //------------------------------------------------------------------------
  /// \ref https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gd228014f19cc0975ebe3e0dd2af6dd1b
  /// \brief Allocates bytes of managed memory on the device.
  //------------------------------------------------------------------------
  handle_malloc(
    cudaMallocManaged(
      reinterpret_cast<void**>(&sequence_length_array_),
      batch_size_ * sizeof(int32_t)));
}

SequenceLengthArray::~SequenceLengthArray()
{
  HandleUnsuccessfulCUDACall handle_free {
    "Failed to free device array for sequencelength array"};

  handle_free(cudaFree(sequence_length_array_));
}

void SequenceLengthArray::copy_host_input_to_device(
  const HostSequenceLengthArray& array)
{
  HandleUnsuccessfulCUDACall handle_copy {
    "Failed to copy values from host to device"};

  handle_copy(cudaMemcpy(
    sequence_length_array_,
    array.sequence_length_array_,
    batch_size_ * sizeof(int32_t),
    cudaMemcpyHostToDevice));
}

} // namespace RecurrentNeuralNetwork