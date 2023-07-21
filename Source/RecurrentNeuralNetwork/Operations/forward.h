#ifndef RECURRENT_NEURAL_NETWORK_OPERATIONS_FORWARD_H
#define RECURRENT_NEURAL_NETWORK_OPERATIONS_FORWARD_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/DataDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "RecurrentNeuralNetwork/WorkAndReserveSpaces.h"

namespace RecurrentNeuralNetwork
{
namespace Operations
{

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNForward
/// \ref 7.2.36. cudnnRNNForward()
/// From API documentation:
/// This routine computes forward response of RNN described by rnnDesc with
/// inputs x, hx, cx, and weights/biases in the weightSpace buffer.
/// RNN outputs are written to y, hy, cy buffers.
/// Note internal RNN signals between time-steps and between layers aren't
/// exposed to user.
/// cudnnStatus_t cudnnRNNForward(
///   cudnnHandle_t handle,
///   cudnnRNNDescriptor_t rnnDesc,
///   cudnnForwardMode_t fwdMode,
///   const int32_t devSeqLengths[],
///   cudnnRNNDataDescriptor_t xDesc,
///   const void* x,
///   cudnnRNNDataDescriptor_t yDesc,
///   const void* y,
///   cudnnTensorDescriptor_t hDesc,
///   const void* hx,
///   void* hy,
///   cudnnTensorDescriptor_t cDesc,
///   const void* cx,
///   void* cy,
/// )
/// When fwdMode is set to CUDNN_FWD_MODE_TRAINING, cudnnRNNForward()
/// function stores intermediate data required to compute first order
/// derivatives in reserve space buffer. Work and reserve space buffer sizes
/// should be computed by cudnnGetRNNTempSpaceSizes() function with same
/// fwdMode setting as used in cudnnRNNForward().
/// cDesc - Input. For LSTM networks only. Tensor descriptor describing the
/// initial or final cell state for LSTM networks only. Cell state data are
/// fully packed. First dimension of tensor depends on dirMode argument
/// passed to cudnnSetRNNDescriptor_v8() call.
/// * if dirMode is CUDNN_UNIDIRECTIONAL, first dimension should match
/// numLayers argument passed to cudnnSetRNNDescriptor_v8().
///
//------------------------------------------------------------------------------

Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall forward(
  DeepNeuralNetwork::CuDNNLibraryHandle& handle,
  RecurrentNeuralNetwork::ManageDescriptor::Descriptor& rnn_descriptor,
  RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array,
  RecurrentNeuralNetwork::ManageDescriptor::DataDescriptor& x_data_descriptor,
  RecurrentNeuralNetwork::ManageDescriptor::DataDescriptor& y_data_descriptor,
  RecurrentNeuralNetwork::WorkAndReserveSpaces& work_and_reserve_spaces);

};

} // namespace Operations
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_OPERATIONS_FORWARD_H