#ifndef ACTIVATION_MANAGE_DESCRIPTOR_SET_DESCRIPTOR_H
#define ACTIVATION_MANAGE_DESCRIPTOR_SET_DESCRIPTOR_H

#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

namespace Activation
{

namespace ManageDescriptor
{

class SetDescriptor
{
  public:

    using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    //--------------------------------------------------------------------------
    /// \ref https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-API.pdf
    /// 3.1.2.1 cudnnActivationMode_t - used to select neuron activation
    /// function.
    /// CUDNN_ACTIVATION_SIGMOID - sigmoid function
    /// S(x) = 1 / (1 + exp(-x))
    /// CUDNN_ACTIVATION_RELU - rectified linear function
    /// f(x) = max(0, x) = (x + |x|) / 2 = x if x > 0, 0 otherwise
    /// CUDNN_ACTIVATION_TANH - hyperbolic tangent function
    /// CUDNN_ACTIVATION_CLIPPED_RELU - clipped rectified linear function
    /// CUDNN_ACTIVATION_ELU - exponential linear function
    /// CUDNN_ACTIVATION_IDENTITY - identity function
    /// CUDNN_ACTIVATION_SWISH - swish function
    /// swish(x) = x / (1 + exp(-beta x))
    ///
    /// 3.1.2.14 cudnnNanPropagation_t - indicates if routine should propagate
    /// Nan numbers.
    /// CUDNN_NOT_PROPAGATE_NAN - Nan numbers are not propagated
    /// CUDNN_PROPAGATE_NAN
    ///
    /// \param clipping_threshold - When activation mode is set to
    /// CUDNN_ACTIVATION_CLIPPED_RELU, this input specified clipping threshold,
    /// and when activation mode is CUDNN_ACTIVATION_RELU, this specifies the
    /// upper bound.
    //--------------------------------------------------------------------------
    SetDescriptor(
      const cudnnActivationMode_t mode,
      const double clipping_threshold = 0.0,
      const cudnnNanPropagation_t option = CUDNN_NOT_PROPAGATE_NAN);

    ~SetDescriptor() = default;

    HandleUnsuccessfulCuDNNCall set_descriptor(
      cudnnActivationDescriptor_t& activation_descriptor);

    HandleUnsuccessfulCuDNNCall set_descriptor(
      ActivationDescriptor& activation_descriptor);

    //--------------------------------------------------------------------------
    //--------------------------------------------------------------------------

    const double clipping_threshold_;
    const cudnnActivationMode_t mode_;
    const cudnnNanPropagation_t option_;
};

} // namespace ManageDescriptor
} // namespace Activation

#endif // ACTIVATION_MANAGE_DESCRIPTOR_SET_DESCRIPTOR_H