#ifndef DEEP_NEURAL_NETWORK_GET_CUDNN_VERSION_H
#define DEEP_NEURAL_NETWORK_GET_CUDNN_VERSION_H

#include <cstddef>

namespace DeepNeuralNetwork
{

class GetCuDNNVersion
{
  public:

    GetCuDNNVersion();

    ~GetCuDNNVersion() = default;

    void pretty_print();

    std::size_t version_number_;

    int major_version_;
    int minor_version_;
    int patch_level_;
};

} // namespace DeepNeuralNetwork

#endif // DEEP_NEURAL_NETWORK_GET_CUDNN_VERSION_H