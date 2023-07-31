#ifndef RECURRENT_NEURAL_NETWORK_OPERATIONS_BACKWARD_ON_WEIGHTS_H
#define RECURRENT_NEURAL_NETWORK_OPERATIONS_BACKWARD_ON_WEIGHTS_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/HiddenDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/InputDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/OutputDescriptor.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
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
/// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNBackwardWeights_v8
/// \ref 8.2.22. cudnnRNNBackwardWeights_v8()
/// cudnnRNNBackwardWeights_v8() computes exact, first-order derivatives of the
/// RNN model with respect to all trainable parameters, weights and biases.
/// cudnnStatus_t cudnnRNNBackwardWeights_v8(
///   cudnnHandle_t handle,
///   cudnnRNNDescriptor_t rnnDesc,
///   cudnnWgradMode_t addGrad,
///   const int32_t devSeqLengths[],
///   cudnnRNNDataDescriptor_t xDesc,
///   const void* x,
///   cudnnTensorDescriptor_t hDesc,
///   const void* hx,
///   cudnnRNNDataDescriptor_t yDesc,
///   const void* y,
///   size_t weightSpaceSize,
///   void* dweightSpace,
///   size_t workSpaceSize,
///   void* workSpace,
///   size_t reserveSpaceSize,
///   void* reserveSpace
/// );
/// \param addGrad - Input. Weight gradient output mode. Currently, only
/// CUDNN_WGRAD_MODE_ADD supported by cudnnRNNBackwardWeights_v8()
/// \param dweightSpace - Output. Address of weight space buffer in GPU memory.
/// \param workSpace - Input/Output. Address of workspace buffer in GPU memory
/// to store temporary data.
/// \param reserveSpace - Input/Output Address of reserve-space buffer in GPU
/// memory.
//------------------------------------------------------------------------------

template <typename T, std::size_t N>
class BackwardOnWeights
{
  public:

    BackwardOnWeights(cudnnWgradMode_t add_grad = CUDNN_WGRAD_MODE_ADD):
      add_grad_{add_grad}
    {}

    ~BackwardOnWeights() = default;

    //--------------------------------------------------------------------------
    /// \param [out] weight_space - d_weight_space has all gradient results with
    /// respect to weights and biases.
    /// \param [in,out] work_and_reserve_spaces - Stores temporary data.
    //--------------------------------------------------------------------------
    Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall backward_on_weights(    
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      RecurrentNeuralNetwork::ManageDescriptor::Descriptor& rnn_descriptor,
      RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array,
      RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor& x_descriptor,
      const RecurrentNeuralNetwork::Modules::Input<T>& x,
      RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<N>&
        h_descriptor,
      const RecurrentNeuralNetwork::Modules::Hidden<T>& hx,
      RecurrentNeuralNetwork::ManageDescriptor::OutputDescriptor& y_descriptor,
      const RecurrentNeuralNetwork::Modules::Output<T>& y,
      RecurrentNeuralNetwork::WeightSpace& weight_space,
      RecurrentNeuralNetwork::WorkAndReserveSpaces& work_and_reserve_spaces)
  {
    Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall handle_backward {
      "Failed to run backward weights operation"};

    handle_backward(
      cudnnRNNBackwardWeights_v8(
        handle.handle_,
        rnn_descriptor.descriptor_,
        add_grad_,
        sequence_length_array.sequence_length_array_,
        x_descriptor.x_data_descriptor_.descriptor_,
        x.x_,
        h_descriptor.descriptor_.descriptor_,
        hx.h_,
        y_descriptor.y_data_descriptor_.descriptor_,
        y.y_,
        weight_space.get_weight_space_size(),
        weight_space.d_weight_space_,
        work_and_reserve_spaces.get_work_space_size(),
        work_and_reserve_spaces.work_space_,
        work_and_reserve_spaces.get_reserve_space_size(),
        work_and_reserve_spaces.reserve_space_));

    return handle_backward;
  }

    Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall backward_on_weights(    
      RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN&
        library_handle_dropout_rnn,
      RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array,
      RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor& x_descriptor,
      const RecurrentNeuralNetwork::Modules::Input<T>& x,
      RecurrentNeuralNetwork::ManageDescriptor::HiddenDescriptor<N>&
        h_descriptor,
      const RecurrentNeuralNetwork::Modules::Hidden<T>& hx,
      RecurrentNeuralNetwork::ManageDescriptor::OutputDescriptor& y_descriptor,
      const RecurrentNeuralNetwork::Modules::Output<T>& y,
      RecurrentNeuralNetwork::WeightSpace& weight_space,
      RecurrentNeuralNetwork::WorkAndReserveSpaces& work_and_reserve_spaces)
  {
    return backward_on_weights(
      library_handle_dropout_rnn.handle_,
      library_handle_dropout_rnn.descriptor_,
      sequence_length_array,
      x_descriptor,
      x,
      h_descriptor,
      hx,
      y_descriptor,
      y,
      weight_space,
      work_and_reserve_spaces);
  }

  private:

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnWgradMode_t
    /// \ref 8.1.1.2. cudnnWgradMode_t - cudnnWgradMode_t is an enumerated type
    /// that selects how buffers holding gradients of loss function, computed
    /// with respect to trainable parameters, are updated.
    /// CUDNN_WGRAD_MODE_ADD - a weight gradient component corresponding to a
    /// new batch of inputs is added to previously evaluated weight gradients.
    /// Before using this mode, buffer holding weight gradients should be
    /// initialized to 0.
    //--------------------------------------------------------------------------
    cudnnWgradMode_t add_grad_;
};

} // namespace Operations
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_OPERATIONS_BACKWARD_ON_WEIGHTS_H