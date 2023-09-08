#include "RecurrentNeuralNetwork/ManageDescriptor/InputDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/OutputDescriptor.h"
#include "RecurrentNeuralNetwork/Modules/Hidden.h"
#include "RecurrentNeuralNetwork/Modules/Input.h"
#include "RecurrentNeuralNetwork/Modules/Output.h"
#include "RecurrentNeuralNetwork/Operations/forward.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "RecurrentNeuralNetwork/WeightSpace.h"
#include "RecurrentNeuralNetwork/WorkAndReserveSpaces.h"
#include "gtest/gtest.h"

#include <cudnn.h>

using RecurrentNeuralNetwork::DefaultParameters;
using RecurrentNeuralNetwork::HostSequenceLengthArray;
using RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor3Dim;
using RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor;
using RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN;
using RecurrentNeuralNetwork::ManageDescriptor::OutputDescriptor;
using RecurrentNeuralNetwork::Modules::Hidden;
using RecurrentNeuralNetwork::Modules::Input;
using RecurrentNeuralNetwork::Modules::Output;
using RecurrentNeuralNetwork::Operations::forward;
using RecurrentNeuralNetwork::SequenceLengthArray;
using RecurrentNeuralNetwork::WeightSpace;
using RecurrentNeuralNetwork::WorkAndReserveSpaces;

namespace GoogleUnitTests
{
namespace Networks
{
namespace RecurrentNeuralNetworks
{
namespace LSTM
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LSTMTests, SetupStepsExplicitlyForDigitRecognizer)
{
  DefaultParameters parameters {};

  parameters.cell_mode_ = CUDNN_LSTM;
  // Row dimension of a 28x28 image.
  parameters.input_size_ = 28;
  parameters.hidden_size_ = 100;
  parameters.projection_size_ = 10;
  parameters.number_of_layers_ = 1;
  parameters.maximum_sequence_length_ = 28;
  parameters.batch_size_ = 100;

  // Creates a N=100=batch size array, each with value T=28 to say each (i.e. a
  // single) batch has 100 samples, each sample a sequence of length (or i.e.
  // time) T = 28.
  SequenceLengthArray sequence_length_array {parameters};
  HostSequenceLengthArray host_array {parameters};
  host_array.set_all_to_maximum_sequence_length();
  sequence_length_array.copy_host_input_to_device(host_array);

  // dataType, layout, maxSeqLength, batchSize and seqLengthArray must match
  // that of yDesc.
  // vectorSize must match inputSize. Use of parameters ensures this.
  InputDescriptor x_descriptor {parameters, sequence_length_array};


  // Default sets dropout probabilty to 0 if not specified in construction.
  // Uses parameters to set RNN descriptor, according to
  // cudnnSetRNNDescriptor_v8(..) signature.
  LibraryHandleDropoutRNN descriptors {parameters};

/*
  WeightSpace weight_space {descriptors};
*/

}

} // namespace LSTM

} // namespace RecurrentNeuralNetwork
} // namespace Networks
} // namespace GoogleUnitTests