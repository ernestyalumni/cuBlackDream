#ifndef RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_INPUT_DESCRIPTOR_H
#define RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_INPUT_DESCRIPTOR_H

#include "RecurrentNeuralNetwork/ManageDescriptor/DataDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"

#include <cudnn.h>

namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

struct InputDescriptor
{
  //----------------------------------------------------------------------------
  /// \ref https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNDataLayout_t
  /// 7.1.2.6. cudnnRNNDataLayout_t
  /// CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED
  /// Data layout is padded, with outer stride from 1 time-step to the next.
  /// CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED
  /// Sequence length is sorted and packed as in basic RNN API.
  /// CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED
  /// Data layout is padded, with outer stride from one batch to the next.
  //----------------------------------------------------------------------------
  InputDescriptor(
    RecurrentNeuralNetwork::Parameters& parameters,
    RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array);

  ~InputDescriptor() = default;

  DataDescriptor x_data_descriptor_;
};

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_INPUT_DESCRIPTOR_H