#ifndef RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_SET_DROPOUT_DESCRIPTOR_H
#define RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_SET_DROPOUT_DESCRIPTOR_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "DropoutDescriptor.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

namespace RecurrentNeuralNetwork
{

namespace ManageDescriptor
{

class SetDropoutDescriptor
{
  public:

    using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    SetDropoutDescriptor(
      const float dropout_probability,
      const unsigned long long seed = 1337ull);

    ~SetDropoutDescriptor() = default;

    //--------------------------------------------------------------------------
    /// \ref 3.2.82. cudnnSetDropoutDescriptor()
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetDropoutDescriptor
    /// cudnnStatus_t cudnnSetDropoutDescriptor(
    ///   cudnnDropoutDescriptor_t dropoutDesc,
    ///   cudnnHandle_t handle,
    ///   float dropout,
    ///   void* states,
    ///   size_t stateSizeInBytes,
    ///   unsigned long long seed)
    /// handle - Input. Handle to previously created cuDNN context.
    /// states - Output. pointer to user-allocated GPU memory that'll hold
    /// random number generator states.
    //--------------------------------------------------------------------------
    HandleUnsuccessfulCuDNNCall set_descriptor(
      DropoutDescriptor& descriptor,
      DeepNeuralNetwork::CuDNNLibraryHandle& handle);

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetDropoutDescriptor
    /// \ref 3.2.82. cudnnSetDropoutDescriptor()
    /// Probability with which value from input is set to zero during dropout
    /// layer.
    //--------------------------------------------------------------------------
    float dropout_;

    // Seed used to initialize random number generator states.
    unsigned long long seed_;
};

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_SET_DROPOUT_DESCRIPTOR_H