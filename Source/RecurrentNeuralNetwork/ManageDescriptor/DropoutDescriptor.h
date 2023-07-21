#ifndef RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_DROPOUT_DESCRIPTOR_H
#define RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_DROPOUT_DESCRIPTOR_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cstddef>
#include <cudnn.h>

namespace RecurrentNeuralNetwork
{

namespace ManageDescriptor
{

class DropoutDescriptor
{
  public:

    using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    DropoutDescriptor();

    ~DropoutDescriptor();

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDropoutGetStatesSize
    /// \ref 3.2.36. cudnnDropoutGetStatesSize()
    /// handle - Input. Handle to previously created cuDNN context.
    /// \brief Query amount of space required to store states of the random
    /// number generators used by cudnnDropoutForward(). 
    //--------------------------------------------------------------------------
    HandleUnsuccessfulCuDNNCall get_states_size_for_forward(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle);

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateDropoutDescriptor
    /// \ref 3.2.9. cudnnCreateDropoutDescriptor()
    //--------------------------------------------------------------------------
    cudnnDropoutDescriptor_t descriptor_;

    // Dropout descriptor parameters.
    std::size_t states_size_;
    void* states_;

    bool is_states_size_known_;
};

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_DROPOUT_DESCRIPTOR_H