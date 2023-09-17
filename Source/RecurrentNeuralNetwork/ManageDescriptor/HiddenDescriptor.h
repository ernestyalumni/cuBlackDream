#ifndef RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_HIDDEN_DESCRIPTOR_H
#define RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_HIDDEN_DESCRIPTOR_H

#include "RecurrentNeuralNetwork/Parameters.h"
#include "Tensors/ManageDescriptor/SetForNDTensor.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cstddef>
#include <cudnn.h>

namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
/// N - rank of the tensor.
//------------------------------------------------------------------------------
template <std::size_t N>
//------------------------------------------------------------------------------
/// TODO: After construction, you must set any other dimensions other than the
/// the first 3, set the strides dimensions, manually, and run, manually,
/// set_descriptor(..) *before* using this in any other function call. Consider
/// automating these steps.
//------------------------------------------------------------------------------
struct HiddenDescriptor
{
  using HandleUnsuccessfulCuDNNCall =
    Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

  HiddenDescriptor():
    descriptor_{},
    set_for_ND_tensor_{}
  {}

  //----------------------------------------------------------------------------
  /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNForward
  /// \ref 7.2.36. cudnnRNNForward()
  /// hDesc - Input. The first dimension of the tensor depends on dirMode passed
  /// to cudnnSetRNNDescriptor_v8().
  /// if dirMode is CUDNN_UNIDIRECTIONAL, then first dimension should match
  /// numLayers passed to cudnnSetRNNDescriptor_v8().
  /// if dirMode is CUDNN_BIDIRECTIONAL, then first dimension should double
  /// numLayers passed to cudnnSetRNNDescriptor_v8().
  /// Second dimension must match batchSize parameter described in xDesc. Third
  /// dimension depends on whether RNN mode is CUDNN_LSTM and whether LSTM
  /// projection is enabled. Specifically,
  /// * if RNN mode is CUDNN_LSTM and LSTM projection enabled, third dimension
  /// must match projSize argument passed to cudnnSetRNNProjectionLayers() call.
  /// * Otherwise, third dimension must match hiddenSize argument passed to
  /// cudnnSetRNNDescriptor_v8() call used to initialize rnnDesc.
  //----------------------------------------------------------------------------
  HiddenDescriptor(const RecurrentNeuralNetwork::Parameters& parameters):
    descriptor_{},
    set_for_ND_tensor_{}
  {
    set_hidden_layers_dimensions_for_forward(parameters);
    set_strides_by_dimensions();
  }  

  ~HiddenDescriptor() = default;

  void set_hidden_layers_dimensions_for_forward(
    const RecurrentNeuralNetwork::Parameters& parameters)
  {
    set_for_ND_tensor_.set_dimensions_array_value(
      0,
      parameters.number_of_layers_ * parameters.get_bidirectional_scale());

    set_for_ND_tensor_.set_dimensions_array_value(
      1,
      parameters.batch_size_);

    if (parameters.cell_mode_ == CUDNN_LSTM)
    {
      //------------------------------------------------------------------------
      /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNBackwardData_v8
      /// \ref 7.2.36. cudnnRNNForward(), hDesc under "Parameters"
      /// hDesc - if RNN mode is CUDNN_LSTM and LSTM projection is enabled, the
      /// third dimension must match the projSize argument passed to the
      /// cudnnSetRNNProjectionLayers() call.
      /// Otherwise, the third dimension must match the hiddenSize argument
      /// passed to cudnnSetRNNDescriptor_v8().
      //------------------------------------------------------------------------
      set_for_ND_tensor_.set_dimensions_array_value(
        2,
        parameters.projection_size_);
    }
    else
    {
      set_for_ND_tensor_.set_dimensions_array_value(
        2,
        parameters.hidden_size_);
    }
  }

  // TODO: Make this more robust, e.g. what if a dimension element was set to 0?
  void set_strides_by_dimensions()
  {
    set_for_ND_tensor_.set_strides_array_value(N - 1, 1);

    int product_of_dimensions {1};

    for (std::size_t i {N - 1}; i > 0; --i)
    {
      product_of_dimensions *=
        set_for_ND_tensor_.get_dimensions_array_value(i);
      set_for_ND_tensor_.set_strides_array_value(
        i - 1,
        product_of_dimensions);
    }
  }

  HandleUnsuccessfulCuDNNCall set_descriptor(
    const RecurrentNeuralNetwork::Parameters& parameters)
  {
    return set_for_ND_tensor_.set_descriptor(descriptor_, parameters);
  }

  Tensors::ManageDescriptor::TensorDescriptor descriptor_;
  Tensors::ManageDescriptor::SetForNDTensor<N> set_for_ND_tensor_;
};

struct HiddenDescriptor3Dim
{
  HiddenDescriptor3Dim(const RecurrentNeuralNetwork::Parameters& parameters);

  ~HiddenDescriptor3Dim() = default;

  Tensors::ManageDescriptor::TensorDescriptor descriptor_;
  Tensors::ManageDescriptor::SetFor3DTensor set_for_3D_tensor_;
};

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_HIDDEN_DESCRIPTOR_H