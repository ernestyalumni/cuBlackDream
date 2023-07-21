#ifndef RECURRENT_NEURAL_NETWORK_WORK_AND_RESERVE_SPACES_H
#define RECURRENT_NEURAL_NETWORK_WORK_AND_RESERVE_SPACES_H

#include "DeepNeuralNetwork/CuDNNLibraryHandle.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/DataDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/Descriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/InputDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/LibraryHandleDropoutRNN.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cstddef>
#include <cudnn.h>

namespace RecurrentNeuralNetwork
{

class WorkAndReserveSpaces
{
  public:

    WorkAndReserveSpaces(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      RecurrentNeuralNetwork::ManageDescriptor::Descriptor& descriptor,
      RecurrentNeuralNetwork::ManageDescriptor::DataDescriptor&
        data_descriptor,
      cudnnForwardMode_t forward_mode = CUDNN_FWD_MODE_TRAINING);

    WorkAndReserveSpaces(
      RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN&
        descriptors,
      RecurrentNeuralNetwork::ManageDescriptor::DataDescriptor&
        data_descriptor,
      cudnnForwardMode_t forward_mode = CUDNN_FWD_MODE_TRAINING);

    WorkAndReserveSpaces(
      RecurrentNeuralNetwork::ManageDescriptor::LibraryHandleDropoutRNN&
        descriptors,
      RecurrentNeuralNetwork::ManageDescriptor::InputDescriptor&
        x_data_descriptor,
      cudnnForwardMode_t forward_mode = CUDNN_FWD_MODE_TRAINING);

    ~WorkAndReserveSpaces();

    inline std::size_t get_work_space_size() const
    {
      return work_space_size_;
    }

    inline float get_work_space_size_in_MiB() const
    {
      return static_cast<float>(work_space_size_) / 1024.0 / 1024.0;
    }

    inline std::size_t get_reserve_space_size() const
    {
      return reserve_space_size_;
    }

    inline float get_reserve_space_size_in_MiB() const
    {
      return static_cast<float>(reserve_space_size_) / 1024.0 / 1024.0;
    }

    inline cudnnForwardMode_t get_forward_mode() const
    {
      return forward_mode_;
    }

    friend Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall forward(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      RecurrentNeuralNetwork::ManageDescriptor::Descriptor& rnn_descriptor,
      RecurrentNeuralNetwork::SequenceLengthArray& sequence_length_array,
      RecurrentNeuralNetwork::ManageDescriptor::DataDescriptor&
        x_data_descriptor,
      RecurrentNeuralNetwork::ManageDescriptor::DataDescriptor&
        y_data_descriptor,
      WorkAndReserveSpaces& work_and_reserve_spaces
      );

  private:

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNTempSpaceSizes
    /// \ref 7.2.29. cudnnGetRNNTempSpaceSizes()
    /// cudnnStatus_t cudnnGetRNNTempSpaceSizes(
    ///   cudnnHandle_t handle,
    ///   cudnnRNNDescriptor_t rnnDesc,
    ///   cudnnForwardMode_t fMode,
    ///   cudnnRNNDataDescriptor_t xDesc,
    ///   ..)
    /// xDesc - Input. Single RNN data descriptor that specifies current RNN
    /// data dimensions: maxSeqLength and batchSize.
    //--------------------------------------------------------------------------
    Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall get_temp_space_sizes(
      DeepNeuralNetwork::CuDNNLibraryHandle& handle,
      RecurrentNeuralNetwork::ManageDescriptor::Descriptor& descriptor,
      RecurrentNeuralNetwork::ManageDescriptor::DataDescriptor&
        data_descriptor);

    void* work_space_;
    void* reserve_space_;

    std::size_t work_space_size_;
    std::size_t reserve_space_size_;

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnForwardMode_t
    /// \ref 7.1.2.2. cudnnForwardMode_t
    /// Specifies inference or training mode in RNN API. This parameter allows
    /// cuDNN library to tune more precisely size of workspace buffer that could
    /// be different in inference and training regimens.
    /// CUDNN_FWD_MODE_INFERENCE - selects inference mode.
    /// CUDNN_FWD_MODE_TRAINING - selects training mode.
    //--------------------------------------------------------------------------
    cudnnForwardMode_t forward_mode_;
};

} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_WORK_AND_RESERVE_SPACES_H