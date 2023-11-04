#ifndef NETWORKS_RECURRENT_NEURAL_NETWORKS_LSTM_LSTM_H
#define NETWORKS_RECURRENT_NEURAL_NETWORKS_LSTM_LSTM_H

#include "RecurrentNeuralNetwork/Operations/forward.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "Setup.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cstddef>
#include <cudnn.h>

namespace Networks
{
namespace RecurrentNeuralNetworks
{

template <typename T = float, std::size_t R = 3>
struct LSTM
{
  LSTM(
    RecurrentNeuralNetwork::LSTMDefaultParameters parameters,
    RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array,
    const cudnnForwardMode_t forward_mode = CUDNN_FWD_MODE_TRAINING
    ):
    setup_{parameters, sequence_length_array, forward_mode}
  {}

  Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall forward()
  {
    return RecurrentNeuralNetwork::Operations::forward<T, R>(
      setup_.descriptors_,
      setup_.x_descriptor_,
      setup_.x_,
      setup_.y_descriptor_,
      setup_.y_,
      setup_.h_descriptor_,
      setup_.hx_,
      setup_.hy_,
      setup_.c_descriptor_,
      setup_.cx_,
      setup_.cy_,
      setup_.weight_space_,
      setup_.spaces_);
  } 

  Setup<T, R> setup_;
};

} // namespace RecurrentNeuralNetworks
} // namespace Networks

#endif // NETWORKS_RECURRENT_NEURAL_NETWORKS_LSTM_LSTM_H