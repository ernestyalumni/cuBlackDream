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
  //----------------------------------------------------------------------------
  InputDescriptor(
    RecurrentNeuralNetwork::Parameters& parameters,
    RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array,
    const cudnnRNNDataLayout_t layout =
      CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED);

  ~InputDescriptor() = default;

  DataDescriptor x_data_descriptor_;
};

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_INPUT_DESCRIPTOR_H