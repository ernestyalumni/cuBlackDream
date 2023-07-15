#include "GetCuDNNVersion.h"

using DeepNeuralNetwork::GetCuDNNVersion;

int main()
{
  GetCuDNNVersion version {};

  version.pretty_print();
}