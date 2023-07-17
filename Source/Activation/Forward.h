#ifndef ACTIVATION_FORWARD_H
#define ACTIVATION_FORWARD_H

#include "Activation/ManageDescriptor/ActivationDescriptor.h"
#include "Algebra/Modules/Tensors/Tensor4D.h"
#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "ManageDescriptor/ActivationDescriptor.h"
#include "Tensors/ManageDescriptor/TensorDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

namespace Activation
{

template <typename T = float>
class Forward
{
  public:

    using ActivationDescriptor =
      Activation::ManageDescriptor::ActivationDescriptor;
    using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    using TensorDescriptor = Tensors::ManageDescriptor::TensorDescriptor;
    using Tensor4D = Algebra::Modules::Tensors::Tensor4D<T>;

    Forward(
      const T alpha = static_cast<T>(1),
      const T beta = static_cast<T>(0)
      ):
      alpha_{alpha},
      beta_{beta}
    {}

    ~Forward() = default;

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnActivationForward
    /// \ref 3.2.1. cudnnActivationForward()
    /// Applies a specified neuron activation function element-wise over each
    /// input value.
    /// In-place operation is allowed, meaning (void*) x, y pointers maybe
    /// equal.
    /// Returns CUDNN_STATUS_SUCCESS, CUDNN_STATUS_NOT_SUPPORTED (function
    /// doesn't support provided configuration), CUDNN_STATUS_BAD_PARAM,
    /// CUDNN_STATUS_EXECUTION_FAILED - function failed to launch on GPU.
    //--------------------------------------------------------------------------
    HandleUnsuccessfulCuDNNCall activation_forward(
      cudnnHandle_t& handle,
      cudnnActivationDescriptor_t& activation_descriptor,
      const cudnnTensorDescriptor_t& x_descriptor,
      const Tensor4D& x_tensor,
      const cudnnTensorDescriptor_t& y_descriptor,
      Tensor4D& y_tensor)
    {
      HandleUnsuccessfulCuDNNCall handle_activation_forward {
        "Failed to run activation forward"};

      handle_activation_forward(
        cudnnActivationForward(
          handle,
          activation_descriptor,
          reinterpret_cast<const void*>(alpha_),
          x_descriptor,
          reinterpret_cast<const void*>(x_tensor.values_),
          reinterpret_cast<const void*>(beta_),
          y_descriptor,
          reinterpret_cast<void*>(y_tensor.begin())));

      return handle_activation_forward;
    }

    HandleUnsuccessfulCuDNNCall activation_forward(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      ActivationDescriptor& activation_descriptor,
      const TensorDescriptor& x_descriptor,
      const Tensor4D& x_tensor,
      const TensorDescriptor& y_descriptor,
      Tensor4D& y_tensor)
    {
      return activation_forward(
        handle.handle_,
        activation_descriptor.descriptor_,
        x_descriptor.descriptor_,
        x_tensor,
        y_descriptor.descriptor_,
        y_tensor);
    }

    HandleUnsuccessfulCuDNNCall inplace_activation_forward(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      ActivationDescriptor& activation_descriptor,
      const TensorDescriptor& x_descriptor,
      Tensor4D& x_tensor)
    {
      return activation_forward(
        handle.handle_,
        activation_descriptor.descriptor_,
        x_descriptor.descriptor_,
        x_tensor,
        x_descriptor.descriptor_,
        x_tensor);      
    }

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnActivationForward
    /// \ref 3.2.1. cudnnActivationForward()
    /// alpha, beta are pointers to scaling factors (in host memory) used to
    /// blend computation result with prior value in output layer.
    /// dstValue = alpha[0] * result + beta[0] * priorDstValue
    //--------------------------------------------------------------------------
    T alpha_[1];
    T beta_[1];
};

} // namespace Activation

#endif // ACTIVATION_FORWARD_H