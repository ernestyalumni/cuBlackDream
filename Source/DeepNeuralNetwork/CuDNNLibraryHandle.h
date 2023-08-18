#ifndef DEEP_NEURAL_NETWORK_CUDNN_LIBRARY_HANDLE_H
#define DEEP_NEURAL_NETWORK_CUDNN_LIBRARY_HANDLE_H

#include <cudnn.h>

namespace DeepNeuralNetwork
{

class CuDNNLibraryHandle
{
  public:

    //--------------------------------------------------------------------------
    /// \ref 3.2.5. cudnnCreate()
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreate
    /// \details Also calls cudnnCreate(), a function that initializes cuDNN
    /// library and creates a handle to an opaque structure holding cuDNN
    /// library context.
    /// cudnnCreate() allocates some internal resources.
    //--------------------------------------------------------------------------
    CuDNNLibraryHandle();

    //--------------------------------------------------------------------------
    /// \ref 3.2.20. cudnnDestroy()
    /// This function releases resources used by cuDNN handle. Because
    /// cudnnCreate() allocates internal resources, release of those resources
    /// by calling cudnnDestroy() will implicitly call cudaDeviceSynchronize().
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnDestroy
    /// It's recommended best practice to call this outside of perforance-
    /// critical code paths.
    //-------------------------------------------------------------------------- 
    ~CuDNNLibraryHandle();

    cudnnHandle_t handle_;
};

} // namespace DeepNeuralNetwork

#endif // DEEP_NEURAL_NETWORK_CUDNN_LIBRARY_HANDLE_H