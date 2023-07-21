#ifndef RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_DATA_DESCRIPTOR_H
#define RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_DATA_DESCRIPTOR_H

#include <cudnn.h>

namespace RecurrentNeuralNetwork
{

namespace ManageDescriptor
{

class DataDescriptor
{
  public:

    DataDescriptor();

    ~DataDescriptor();

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateRNNDescriptor
    /// \ref 7.2.6.
    /// \details cudnnRNNDescriptor_t is a pointer to an opaque structure
    /// holding the description of an RNN operation.
    //--------------------------------------------------------------------------
    cudnnRNNDataDescriptor_t descriptor_;
};

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_DATA_DESCRIPTOR_H