#ifndef RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_DESCRIPTOR_H
#define RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_DESCRIPTOR_H

#include "DropoutDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

namespace RecurrentNeuralNetwork
{

namespace ManageDescriptor
{

class Descriptor
{
  public:

    using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    friend HandleUnsuccessfulCuDNNCall set_rnn_descriptor(
      Descriptor&,
      RecurrentNeuralNetwork::Parameters&,
      DropoutDescriptor&);

    Descriptor();

    ~Descriptor();

    HandleUnsuccessfulCuDNNCall get_RNN_parameters(
      RecurrentNeuralNetwork::Parameters& parameters,
      DropoutDescriptor& dropout_descriptor);

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateRNNDescriptor
    /// \ref 7.2.6.
    /// \details cudnnRNNDescriptor_t is a pointer to an opaque structure
    /// holding the description of an RNN operation.
    //--------------------------------------------------------------------------
    cudnnRNNDescriptor_t descriptor_;

    bool is_set_;
};

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_DESCRIPTOR_H