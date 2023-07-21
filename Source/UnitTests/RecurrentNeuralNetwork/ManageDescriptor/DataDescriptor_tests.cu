#include "RecurrentNeuralNetwork/ManageDescriptor/DataDescriptor.h"
#include "gtest/gtest.h"

using RecurrentNeuralNetwork::ManageDescriptor::DataDescriptor;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DataDescriptor, DefaultConstructs)
{
  DataDescriptor data_descriptor {};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(DataDescriptor, Destructs)
{
  {
    DataDescriptor data_descriptor {};

    SUCCEED();
  }
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests