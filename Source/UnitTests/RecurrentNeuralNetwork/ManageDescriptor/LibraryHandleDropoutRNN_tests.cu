#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "gtest/gtest.h"

#include <cudnn.h>
#include <stdexcept>

using RecurrentNeuralNetwork::DefaultParameters;
using RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN;
using std::runtime_error;

namespace GoogleUnitTests
{
namespace RecurrentNeuralNetwork
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(LibraryHandleDropoutRNNTests, FailsOnProjectionSizeLargerThanHiddenSize)
{
  DefaultParameters parameters {};
  parameters.projection_size_ = 600;

  EXPECT_THROW({
    try
    {
      LibraryHandleDropoutRNN descriptors {parameters};
    }
    catch(const runtime_error& err)
    {
      EXPECT_STREQ(
        "Failed to set RNN descriptor in LibraryHandleDropoutRNN",
        err.what());
      throw;
    }
  },
  runtime_error);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  LibraryHandleDropoutRNNTests,
  FailsOnProjectionSizeSmallerThanHiddenSizeWhenCellModeIsCUDNN_RNN_RELU)
{
  DefaultParameters parameters {};
  parameters.projection_size_ = 500;

  EXPECT_THROW({
    try
    {
      LibraryHandleDropoutRNN descriptors {parameters};
    }
    catch(const runtime_error& err)
    {
      EXPECT_STREQ(
        "Failed to set RNN descriptor in LibraryHandleDropoutRNN",
        err.what());
      throw;
    }
  },
  runtime_error);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  LibraryHandleDropoutRNNTests,
  FailsOnProjectionSizeLargerThanHiddenSizeWhenCellModeIsCUDNN_LSTM)
{
  DefaultParameters parameters {};
  parameters.cell_mode_ = CUDNN_LSTM;
  parameters.projection_size_ = 600;

  EXPECT_THROW({
    try
    {
      LibraryHandleDropoutRNN descriptors {parameters};
    }
    catch(const runtime_error& err)
    {
      EXPECT_STREQ(
        "Failed to set RNN descriptor in LibraryHandleDropoutRNN",
        err.what());
      throw;
    }
  },
  runtime_error);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  LibraryHandleDropoutRNNTests,
  ConstructssOnProjectionSizeSmallerThanHiddenSizeWhenCellModeIsCUDNN_LSTM)
{
  DefaultParameters parameters {};
  parameters.cell_mode_ = CUDNN_LSTM;
  parameters.projection_size_ = 400;

  LibraryHandleDropoutRNN descriptors {parameters};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  LibraryHandleDropoutRNNTests,
  ConstructssOnProjectionSizeEqualToHiddenSizeWhenCellModeIsCUDNN_LSTM)
{
  DefaultParameters parameters {};
  parameters.cell_mode_ = CUDNN_LSTM;

  LibraryHandleDropoutRNN descriptors {parameters};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  LibraryHandleDropoutRNNTests,
  ConstructssOnHiddenSizeLargerThanInputSizeIfProjectionSizeIsEqual)
{
  DefaultParameters parameters {};
  parameters.hidden_size_ = 600;
  parameters.projection_size_ = 600;

  LibraryHandleDropoutRNN descriptors {parameters};

  SUCCEED();
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(
  LibraryHandleDropoutRNNTests,
  FailsOnProjectionSizeSmallerThanLargerHiddenSizeWhenCellModeIsCUDNN_RNN_RELU)
{
  DefaultParameters parameters {};
  parameters.hidden_size_ = 600;
  parameters.projection_size_ = 599;

  EXPECT_THROW({
    try
    {
      LibraryHandleDropoutRNN descriptors {parameters};
    }
    catch(const runtime_error& err)
    {
      EXPECT_STREQ(
        "Failed to set RNN descriptor in LibraryHandleDropoutRNN",
        err.what());
      throw;
    }
  },
  runtime_error);
}

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork
} // namespace GoogleUnitTests