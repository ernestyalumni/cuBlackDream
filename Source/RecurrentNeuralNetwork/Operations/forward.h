#ifndef RECURRENT_NEURAL_NETWORK_OPERATIONS_FORWARD_H
#define RECURRENT_NEURAL_NETWORK_OPERATIONS_FORWARD_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/HiddenDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/InputDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/OutputDescriptor.h"
#include "RecurrentNeuralNetwork/Modules/Hidden.h"
#include "RecurrentNeuralNetwork/Modules/Input.h"
#include "RecurrentNeuralNetwork/Modules/Output.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "RecurrentNeuralNetwork/WeightSpace.h"
#include "RecurrentNeuralNetwork/WorkAndReserveSpaces.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cstddef>
#include <cudnn.h>

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
///   void* y,
///   cudnnTensorDescriptor_t hDesc,
///   const void* hx,
///   void* hy,
///   cudnnTensorDescriptor_t cDesc,
///   const void* cx,
///   void* cy,
///   size_t weightSpaceSize,
///   const void* weightSpace,
///   size_t workSpaceSize,
///   void* workSpace,
///   size_t reserveSpaceSize,
///   void* reserveSpace;
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
/// \param [in] xDesc -
/// dataType, layout, maxSeqLength, batchSize, and seqLengthArray must match
/// that of yDesc. vectorSize must match inputSize argument passed to
/// cudnnSetRNNDescriptor_v8().
/// \param yDesc - Input. Previously initialized RNN data descriptor. The
/// dataType, layout, maxSeqLength, batchSize, seqLengthArray must match that of
/// xDesc. Parameter vectorSize depends on whether LSTM projection is enabled
/// and whether netwwork is bi-directional.
/// \param [out] y
/// \param hx - Input. Pointer to GPU buffer with RNN initial hidden state.
/// Data dimensions described by hDesc tensor descriptor.
/// \param [out] hy - Pointer to GPU buffer where final RNN hidden state
/// should be stored. Data dimensions are described by hDesc tensor descriptor.
/// \param [out] cy - For LSTM networks only. Pointer to GPU buffer where final
/// LSTM state data should be stored. Data dimensions described by cDesc tensor
/// descriptor. If NULL pointer passed, final LSTM cell state won't be saved.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
/// N - dimension of the tensor.
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall forward(
  DeepNeuralNetwork::CuDNNLibraryHandle& handle,
  RecurrentNeuralNetwork::ManageDescriptor::Descriptor& rnn_descriptor,
  RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array,
  RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor& x_descriptor,
  const RecurrentNeuralNetwork::Modules::Input<T>& x,
  RecurrentNeuralNetwork::ManageDescriptor::OutputDescriptor& y_descriptor,
  RecurrentNeuralNetwork::Modules::Output<T>& y,
  RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<N>& h_descriptor,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& hx,
  RecurrentNeuralNetwork::Modules::Hidden<T>& hy,
  RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<N>& c_descriptor,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& cx,
  RecurrentNeuralNetwork::Modules::Hidden<T>& cy,
  RecurrentNeuralNetwork::WeightSpace& weight_space,
  RecurrentNeuralNetwork::WorkAndReserveSpaces& work_and_reserve_spaces)
{
  Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall handle_forward {
    "Failed to run forward operation"};

  handle_forward(
    cudnnRNNForward(
      handle.handle_,
      rnn_descriptor.descriptor_,
      work_and_reserve_spaces.get_forward_mode(),
      sequence_length_array.sequence_length_array_,
      x_descriptor.x_data_descriptor_.descriptor_,
      x.x_,
      y_descriptor.y_data_descriptor_.descriptor_,
      y.y_,
      h_descriptor.descriptor_.descriptor_,
      hx.h_,
      hy.h_,
      c_descriptor.descriptor_.descriptor_,
      cx.h_,
      cy.h_,
      weight_space.get_weight_space_size(),
      weight_space.weight_space_,
      work_and_reserve_spaces.get_work_space_size(),
      work_and_reserve_spaces.work_space_,
      work_and_reserve_spaces.get_reserve_space_size(),
      work_and_reserve_spaces.reserve_space_
      ));

  return handle_forward;
}

template <typename T, std::size_t N>
Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall forward(
  RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN&
    library_handle_dropout_rnn,
  RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array,
  RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor& x_descriptor,
  const RecurrentNeuralNetwork::Modules::Input<T>& x,
  RecurrentNeuralNetwork::ManageDescriptor::OutputDescriptor& y_descriptor,
  RecurrentNeuralNetwork::Modules::Output<T>& y,
  RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<N>& h_descriptor,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& hx,
  RecurrentNeuralNetwork::Modules::Hidden<T>& hy,
  RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<N>& c_descriptor,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& cx,
  RecurrentNeuralNetwork::Modules::Hidden<T>& cy,
  RecurrentNeuralNetwork::WeightSpace& weight_space,
  RecurrentNeuralNetwork::WorkAndReserveSpaces& work_and_reserve_spaces)
{
  return forward<T, N>(
    library_handle_dropout_rnn.handle_,
    library_handle_dropout_rnn.descriptor_,
    sequence_length_array,
    x_descriptor,
    x,
    y_descriptor,
    y,
    h_descriptor,
    hx,
    hy,
    c_descriptor,
    cx,
    cy,
    weight_space,
    work_and_reserve_spaces);
}

//------------------------------------------------------------------------------
/// N - dimension of the tensor.
/// \details Not a LSTM (Long Short Term Memory)
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall forward_no_lstm(
  DeepNeuralNetwork::CuDNNLibraryHandle& handle,
  RecurrentNeuralNetwork::ManageDescriptor::Descriptor& rnn_descriptor,
  RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array,
  RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor& x_descriptor,
  RecurrentNeuralNetwork::Modules::Input<T>& x,
  RecurrentNeuralNetwork::ManageDescriptor::OutputDescriptor& y_descriptor,
  RecurrentNeuralNetwork::Modules::Output<T>& y,
  RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<N>& h_descriptor,
  RecurrentNeuralNetwork::Modules::Hidden<T>& hx,
  RecurrentNeuralNetwork::Modules::Hidden<T>& hy,
  RecurrentNeuralNetwork::WeightSpace& weight_space,
  RecurrentNeuralNetwork::WorkAndReserveSpaces& work_and_reserve_spaces)
{
  Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall handle_forward {
    "Failed to run forward operation"};

  handle_forward(
    cudnnRNNForward(
      handle.handle_,
      rnn_descriptor.descriptor_,
      work_and_reserve_spaces.get_forward_mode(),
      sequence_length_array.sequence_length_array_,
      x_descriptor.x_data_descriptor_.descriptor_,
      x.x_,
      y_descriptor.y_data_descriptor_.descriptor_,
      y.y_,
      h_descriptor.descriptor_.descriptor_,
      hx.h_,
      hy.h_,
      h_descriptor.descriptor_.descriptor_,
      nullptr,
      nullptr,
      weight_space.get_weight_space_size(),
      weight_space.weight_space_,
      work_and_reserve_spaces.get_work_space_size(),
      work_and_reserve_spaces.work_space_,
      work_and_reserve_spaces.get_reserve_space_size(),
      work_and_reserve_spaces.reserve_space_
      ));

  return handle_forward;
}

} // namespace Operations
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_OPERATIONS_FORWARD_H