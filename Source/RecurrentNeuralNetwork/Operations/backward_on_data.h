#ifndef RECURRENT_NEURAL_NETWORK_OPERATIONS_BACKWARD_ON_DATA_H
#define RECURRENT_NEURAL_NETWORK_OPERATIONS_BACKWARD_ON_DATA_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/CellDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/HiddenDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/InputDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/OutputDescriptor.h"
#include "RecurrentNeuralNetwork/Modules/Cell.h"
#include "RecurrentNeuralNetwork/Modules/Hidden.h"
#include "RecurrentNeuralNetwork/Modules/Input.h"
#include "RecurrentNeuralNetwork/Modules/Output.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "RecurrentNeuralNetwork/WeightSpace.h"
#include "RecurrentNeuralNetwork/WorkAndReserveSpaces.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cstddef> // std::size_t
#include <cudnn.h>

namespace RecurrentNeuralNetwork
{
namespace Operations
{

//------------------------------------------------------------------------------
/// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNBackwardData_v8
/// \ref 8.2.19. cudnnRNNBackwardData_v8()
/// From API documentation:
/// cudnnRNNBackwardData_v8() computes exact, first-order derivatives of RNN
/// model with respect to its inputs x, hx, and for LSTM cell types also cx.
/// cudnnStatus_t cudnnRNNBackwardData_v8(
///   cudnnHandle_t handle,
///   cudnnRNNDescriptor_t rnnDesc,
///   const int32_t devSeqLengths[],
///   cudnnRNNDataDescriptor_t yDesc,
///   const void* y,
///   const void* dy,
///   cudnnRNNDataDescriptor_t xDesc,
///   void* dx,
///   cudnnTensorDescriptor_t hDesc,
///   const void* hx,
///   const void* dhy,
///   void* dhx,
///   cudnnTensorDescriptor_t cDesc,
///   const void* cx,
///   const void* dcy,
///   void* dcx,
///   size_t weightSpaceSize,
///   const void* weightSpace,
///   size_t workSpaceSize,
///   void* workSpace,
///   size_t reserveSpaceSize,
///   void* reserveSpace
/// )
/// \param [out] dx - Data pointer to GPU memory where back-propagated gradients
/// of the loss function with respect to RNN primary input x is stored. Vectors
/// expected to be arranged in memory according to layout specified by xDesc.
/// \param [out] dhx - Pointer to GPU buffer where first-order derivatives
/// corresponding to initial hidden state variables should be stored.
/// \param [in] cx, dcy - Input. For LSTM networks only. Addresses of GPU
/// buffers with initial LSTM state data and gradient deltas dcy.
/// \param [out] dcx - For LSTM networks only. Pointer to GPU buffer where
/// first-order derivatves corresponding to initial LSTM state variables should
/// be stored.
///
/// cudnnRNNBackwardData_v8() must be called after cudnnRNNForward()
//------------------------------------------------------------------------------
template <typename T, std::size_t N>
Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall backward_on_data(
  DeepNeuralNetwork::CuDNNLibraryHandle& handle,
  RecurrentNeuralNetwork::ManageDescriptor::Descriptor& rnn_descriptor,
  RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array,
  RecurrentNeuralNetwork::ManageDescriptor::OutputDescriptor& y_descriptor,
  const RecurrentNeuralNetwork::Modules::Output<T>& y,
  const RecurrentNeuralNetwork::Modules::Output<T>& dy,
  RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor& x_descriptor,
  RecurrentNeuralNetwork::Modules::Input<T>& dx,
  RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<N>& h_descriptor,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& hx,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& dhy,
  RecurrentNeuralNetwork::Modules::Hidden<T>& dhx,
  RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<N>& c_descriptor,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& cx,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& dcy,
  RecurrentNeuralNetwork::Modules::Hidden<T>& dcx,
  RecurrentNeuralNetwork::WeightSpace& weight_space,
  RecurrentNeuralNetwork::WorkAndReserveSpaces& work_and_reserve_spaces)
{
  Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall handle_backward {
    "Failed to run backward data operation"};

  handle_backward(
    cudnnRNNBackwardData_v8(
      handle.handle_,
      rnn_descriptor.descriptor_,
      sequence_length_array.sequence_length_array_,
      y_descriptor.y_data_descriptor_.descriptor_,
      y.y_,
      dy.y_,
      x_descriptor.x_data_descriptor_.descriptor_,
      // Output, grad(l)(x)
      dx.x_,
      h_descriptor.descriptor_.descriptor_,
      hx.h_,
      dhy.h_,
      dhx.h_,
      c_descriptor.descriptor_.descriptor_,
      cx.h_,
      dcy.h_,
      dcx.h_,
      weight_space.get_weight_space_size(),
      weight_space.weight_space_,
      work_and_reserve_spaces.get_work_space_size(),
      work_and_reserve_spaces.work_space_,
      work_and_reserve_spaces.get_reserve_space_size(),
      work_and_reserve_spaces.reserve_space_
      ));

  return handle_backward;
}

template <typename T, std::size_t N>
Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall backward_on_data(
  RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN&
    library_handle_dropout_rnn,
  RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array,
  RecurrentNeuralNetwork::ManageDescriptor::OutputDescriptor& y_descriptor,
  const RecurrentNeuralNetwork::Modules::Output<T>& y,
  const RecurrentNeuralNetwork::Modules::Output<T>& dy,
  RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor& x_descriptor,
  RecurrentNeuralNetwork::Modules::Input<T>& dx,
  RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<N>& h_descriptor,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& hx,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& dhy,
  RecurrentNeuralNetwork::Modules::Hidden<T>& dhx,
  RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<N>& c_descriptor,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& cx,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& dcy,
  RecurrentNeuralNetwork::Modules::Hidden<T>& dcx,
  RecurrentNeuralNetwork::WeightSpace& weight_space,
  RecurrentNeuralNetwork::WorkAndReserveSpaces& work_and_reserve_spaces)
{
  return backward_on_data<T, N>(
    library_handle_dropout_rnn.handle_,
    library_handle_dropout_rnn.descriptor_,
    sequence_length_array,
    y_descriptor,
    y,
    dy,
    x_descriptor,
    dx,
    h_descriptor,
    hx,
    dhy,
    dhx,
    c_descriptor,
    cx,
    dcy,
    dcx,
    weight_space,
    work_and_reserve_spaces);
}

template <typename T, std::size_t N>
Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall backward_on_data(
  RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN&
    library_handle_dropout_rnn,
  RecurrentNeuralNetwork::ManageDescriptor::OutputDescriptor& y_descriptor,
  // Produced by preceding cudnnRNNForward() Call.
  const RecurrentNeuralNetwork::Modules::Output<T>& y,
  const RecurrentNeuralNetwork::Modules::Output<T>& dy,
  RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor& x_descriptor,
  RecurrentNeuralNetwork::Modules::Input<T>& dx,
  RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<N>& h_descriptor,
  // Same as from cudnnRNNForward()
  const RecurrentNeuralNetwork::Modules::Hidden<T>& hx,
  const RecurrentNeuralNetwork::Modules::Hidden<T>& dhy,
  RecurrentNeuralNetwork::Modules::Hidden<T>& dhx,
  RecurrentNeuralNetwork::ManageDescriptor::CellDescriptor<N>& c_descriptor,
  const RecurrentNeuralNetwork::Modules::Cell<T>& cx,
  const RecurrentNeuralNetwork::Modules::Cell<T>& dcy,
  RecurrentNeuralNetwork::Modules::Cell<T>& dcx,
  const RecurrentNeuralNetwork::WeightSpace& weight_space,
  RecurrentNeuralNetwork::WorkAndReserveSpaces& work_and_reserve_spaces)
{
  Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall handle_backward {
    "Failed to run backward data operation"};

  handle_backward(
    cudnnRNNBackwardData_v8(
      library_handle_dropout_rnn.handle_.handle_,
      library_handle_dropout_rnn.descriptor_.descriptor_,
      //------------------------------------------------------------------------
      /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNBackwardData_v8
      /// devSeqLengths [in] In cuDNN 8.9.1 and later, devSeqLengths should be
      /// NULL
      //------------------------------------------------------------------------
      nullptr,
      y_descriptor.y_data_descriptor_.descriptor_,
      y.y_,
      dy.y_,
      x_descriptor.x_data_descriptor_.descriptor_,
      // Output, grad(l)(x)
      dx.x_,
      h_descriptor.descriptor_.descriptor_,
      hx.h_,
      dhy.h_,
      // Output
      dhx.h_,
      c_descriptor.descriptor_.descriptor_,
      cx.c_,
      dcy.c_,
      // Output
      dcx.c_,
      weight_space.get_weight_space_size(),
      weight_space.weight_space_,
      work_and_reserve_spaces.get_work_space_size(),
      // Input/Output
      work_and_reserve_spaces.work_space_,
      work_and_reserve_spaces.get_reserve_space_size(),
      // Input/Output
      work_and_reserve_spaces.reserve_space_
      ));

  return handle_backward;
}

} // namespace Operations
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_OPERATIONS_BACKWARD_ON_DATA_H