#ifndef RECURRENT_NEURAL_NETWORK_WEIGHT_SPACE_H
#define RECURRENT_NEURAL_NETWORK_WEIGHT_SPACE_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cstddef>
#include <cudnn.h>

namespace RecurrentNeuralNetwork
{

class WeightSpace
{
  public:

    WeightSpace(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      RecurrentNeuralNetwork::ManageDescriptor::Descriptor& descriptor);

    WeightSpace(
      RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN&
        descriptors);

    ~WeightSpace();

    inline std::size_t get_weight_space_size() const
    {
      return weight_space_size_;
    }

    inline float get_weight_space_size_in_MiB() const
    {
      return static_cast<float>(weight_space_size_) / 1024.0 / 1024.0;
    }

  private:

    Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall get_weight_space_size(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      RecurrentNeuralNetwork::ManageDescriptor::Descriptor& descriptor);

    void* weight_space_;
    void* d_weight_space_;

    std::size_t weight_space_size_;
};

} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_WEIGHT_SPACE_H