#ifndef RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_LIBRARY_HANDLE_DROPOUT_RNN_H
#define RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_LIBRARY_HANDLE_DROPOUT_RNN_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/DropoutDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"

namespace RecurrentNeuralNetwork
{

namespace ManageDescriptor
{

struct LibraryHandleDropoutRNN
{
  LibraryHandleDropoutRNN(
    const RecurrentNeuralNetwork::Parameters& parameters,
    const float dropout_probability = 0,
    const unsigned long long seed = 1337ull);

  ~LibraryHandleDropoutRNN() = default;

  DeepNeuralNetwork::CuDNNLibraryHandle handle_;
  RecurrentNeuralNetwork::ManageDescriptor::DropoutDescriptor
    dropout_descriptor_;
  RecurrentNeuralNetwork::ManageDescriptor::Descriptor descriptor_;

  float dropout_probability_;
  unsigned long long seed_;
};

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_LIBRARY_HANDLE_DROPOUT_RNN_H