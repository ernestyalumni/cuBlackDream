#include "GetCuDNNVersion.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>
#include <iostream>

using std::cout;

using Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

namespace DeepNeuralNetwork
{

GetCuDNNVersion::GetCuDNNVersion():
  // 3.2.64 cudnnGetVersion(), pp. 62, cudnn_ops_infer.so Library
  // https://docs.nvidia.com/deeplearning/cudnn/pdf/cuDNN-API.pdf
  version_number_{cudnnGetVersion()},
  major_version_{-1},
  minor_version_{-1},
  patch_level_{-1}
{
  HandleUnsuccessfulCuDNNCall property_handle {"Failed to get property"};

  property_handle(cudnnGetProperty(MAJOR_VERSION, &major_version_));
  property_handle(cudnnGetProperty(MINOR_VERSION, &minor_version_));
  property_handle(cudnnGetProperty(PATCH_LEVEL, &patch_level_));
}

void GetCuDNNVersion::pretty_print()
{
  cout << "Version number: " << version_number_ << "\n";
  cout << "Major version: " << major_version_ << "\n";
  cout << "Minor version: " << minor_version_ << "\n";
  cout << "Patch Level: " << patch_level_ << "\n";
}

} // namespace DeepNeuralNetwork