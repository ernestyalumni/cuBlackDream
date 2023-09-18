#ifndef RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_OUTPUT_DESCRIPTOR_H
#define RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_OUTPUT_DESCRIPTOR_H

#include "RecurrentNeuralNetwork/ManageDescriptor/DataDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"

#include <cudnn.h>

namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

struct OutputDescriptor
{
  OutputDescriptor(
    RecurrentNeuralNetwork::Parameters& parameters,
    RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array);

  ~OutputDescriptor() = default;

  DataDescriptor y_data_descriptor_;
};

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_OUTPUT_DESCRIPTOR_H