#ifndef NETWORKS_RECURRENT_NEURAL_NETWORKS_LSTM_SETUP_H
#define NETWORKS_RECURRENT_NEURAL_NETWORKS_LSTM_SETUP_H

#include "RecurrentNeuralNetwork/ManageDescriptor/CellDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/HiddenDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/InputDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/OutputDescriptor.h"
#include "RecurrentNeuralNetwork/Modules/Cell.h"
#include "RecurrentNeuralNetwork/Modules/Hidden.h"
#include "RecurrentNeuralNetwork/Modules/Input.h"
#include "RecurrentNeuralNetwork/Modules/Output.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "RecurrentNeuralNetwork/WeightSpace.h"
#include "RecurrentNeuralNetwork/WorkAndReserveSpaces.h"

#include <cstddef>
#include <cudnn.h>

namespace Networks
{
namespace RecurrentNeuralNetworks
{

//------------------------------------------------------------------------------
/// R - rank of tensors.
//------------------------------------------------------------------------------
template <typename T = float, std::size_t R = 3>
struct Setup
{
  //----------------------------------------------------------------------------
  /// \details TODO: fix const-ness for parameters input.
  //----------------------------------------------------------------------------
  Setup(
    RecurrentNeuralNetwork::LSTMDefaultParameters parameters,
    RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array,
    const cudnnForwardMode_t forward_mode = CUDNN_FWD_MODE_TRAINING
    ):
    parameters_{parameters},
    forward_mode_{forward_mode},
    x_descriptor_{parameters, sequence_length_array},
    y_descriptor_{parameters, sequence_length_array},
    descriptors_{parameters},
    h_descriptor_{parameters},
    c_descriptor_{parameters},
    weight_space_{descriptors_},
    spaces_{descriptors_, x_descriptor_, forward_mode_},
    x_{parameters},
    y_{parameters},
    hx_{parameters},
    hy_{parameters},
    cx_{parameters},
    cy_{parameters}
  {
    h_descriptor_.set_descriptor(parameters_);
    c_descriptor_.set_descriptor(parameters_);
  }

  RecurrentNeuralNetwork::LSTMDefaultParameters parameters_;
  cudnnForwardMode_t forward_mode_;

  RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor x_descriptor_;
  RecurrentNeuralNetwork::ManageDescriptor::OutputDescriptor y_descriptor_;

  RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN
    descriptors_;

  RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<R> h_descriptor_;
  RecurrentNeuralNetwork::ManageDescriptor::CellDescriptor<R> c_descriptor_;

  RecurrentNeuralNetwork::WeightSpace weight_space_;
  RecurrentNeuralNetwork::WorkAndReserveSpaces spaces_;

  RecurrentNeuralNetwork::Modules::Input<T> x_;
  RecurrentNeuralNetwork::Modules::Output<T> y_;

  RecurrentNeuralNetwork::Modules::Hidden<T> hx_;
  RecurrentNeuralNetwork::Modules::Hidden<T> hy_;

  RecurrentNeuralNetwork::Modules::Cell<T> cx_;
  RecurrentNeuralNetwork::Modules::Cell<T> cy_;
};

} // namespace RecurrentNeuralNetworks
} // namespace Networks

#endif // NETWORKS_RECURRENT_NEURAL_NETWORKS_LSTM_SETUP_H