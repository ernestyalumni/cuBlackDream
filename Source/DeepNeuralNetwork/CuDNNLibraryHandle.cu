#include "CuDNNLibraryHandle.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace DeepNeuralNetwork
{

CuDNNLibraryHandle::CuDNNLibraryHandle():
  handle_{}
{
  HandleUnsuccessfulCuDNNCall create_handle {"Failed to create handle"};
  create_handle(cudnnCreate(&handle_));
}

CuDNNLibraryHandle::~CuDNNLibraryHandle()
{
  HandleUnsuccessfulCuDNNCall destroy_handle {"Failed to destroy handle"};

  destroy_handle(cudnnDestroy(handle_));
}

} // namespace DeepNeuralNetwork