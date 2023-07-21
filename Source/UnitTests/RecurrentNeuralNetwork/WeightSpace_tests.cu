#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/WeightSpace.h"

#include "gtest/gtest.h"

using RecurrentNeuralNetwork::DefaultParameters;
using RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN;
using RecurrentNeuralNetwork::WeightSpace;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(WeightSpace, Constructs)
{
  DefaultParameters parameters {};

  LibraryHandleDropoutRNN descriptors {parameters};

  WeightSpace weight_space {descriptors.handle_, descriptors.descriptor_};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(WeightSpace, ConstructsWithLibraryHandleDropoutRNN)
{
  DefaultParameters parameters {};

  LibraryHandleDropoutRNN descriptors {parameters};

  WeightSpace weight_space {descriptors};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(WeightSpace, GetWeightSpaceSizeGetsSize)
{
  DefaultParameters parameters {};

  LibraryHandleDropoutRNN descriptors {parameters};

  WeightSpace weight_space {descriptors};

  EXPECT_EQ(weight_space.get_weight_space_size(), 4202496);
}

} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests