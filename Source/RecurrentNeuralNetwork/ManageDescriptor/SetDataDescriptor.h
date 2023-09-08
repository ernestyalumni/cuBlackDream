#ifndef RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_SET_DATA_DESCRIPTOR_H
#define RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_SET_DATA_DESCRIPTOR_H

#include "DataDescriptor.h"
#include "RecurrentNeuralNetwork/Parameters.h"
#include "RecurrentNeuralNetwork/SequenceLengthArray.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

namespace RecurrentNeuralNetwork
{

namespace ManageDescriptor
{

class SetDataDescriptor
{
  public:

    using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    SetDataDescriptor(
      const cudnnRNNDataLayout_t layout =
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED);

    ~SetDataDescriptor() = default;

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNDataDescriptor
    /// \ref 7.2.47. cudnnSetRNNDataDescriptor()
    /// This data structure is intended to support unpacked (padded) layout for
    /// input; a packed (unpadded) layout is also supported for backward
    /// compatibility.
    /// cudnnStatus_t cudnnSetRNNDataDescriptor(
    ///   cudnnRNNDataDescriptor_t RNNDataDesc,
    ///   cudnnDataType_t dataType,
    ///   cudnnRNNDataLayout_t layout,
    ///   int maxSeqLength,
    ///   int batchSize,
    ///   int vectorSize,
    ///   const int seqLengthArray[],
    ///   void *paddingFill);
    /// vectorSize - vector length (embedding size) of the input or output
    /// tensor at each time-step.
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    /// \details Use parameters to ensure length of this vector is the same as
    /// input size (inputSize).
    //--------------------------------------------------------------------------
    HandleUnsuccessfulCuDNNCall set_descriptor_for_input(
      DataDescriptor& descriptor,
      const Parameters& parameters,
      const SequenceLengthArray& sequence_length_array);

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNForward
    /// \ref 7.2.36. cudnnRNNForward()
    /// \details dataType, layout, maxSeqLength, batchSize, and seqLengthArray
    /// must match that of xDesc.
    /// For uni-direcctional models, vectorSize must match hiddenSize passed to
    /// cudnnSetRNNDescriptor_v8().
    /// If LSTM projection enabled, vectorSize must be same as projSize argument
    /// passed to cudnnSetRNNDescriptor_v8().
    //--------------------------------------------------------------------------
    HandleUnsuccessfulCuDNNCall set_descriptor_for_output(
      DataDescriptor& descriptor,
      const Parameters& parameters,
      const SequenceLengthArray& sequence_length_array);

    //--------------------------------------------------------------------------
    /// The memory layout of the RNN data tensor.
    /// \href https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNDataLayout_t
    /// \ref 7.1.2.6. cudnnRNNDataLayout_t
    /// CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED - data layout is padded, with
    /// outer stride from one time-step to the next. i.e. time major.
    /// CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED - The sequence length is sorted
    /// and packed as in basic RNN API. For backward compatibility, see
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnRNNForwardInferenceEx
    /// 7.2.38. cudnnRNNForwardInferenceEx().
    /// CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED - Data layout is padded, with
    /// outer stride from 1 batch to next. i.e. batch major.
    //--------------------------------------------------------------------------
    cudnnRNNDataLayout_t layout_;

  private:

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetRNNDataDescriptor
    /// \ref 7.2.47. cudnnSetRNNDataDescriptor()
    /// User-defined symbol for filling padding position in RNN output. This is
    /// only effective when descriptor is describing RNN output, and unpacked
    /// layer is specified. Symbol should be in host memory, and is interpreted
    /// as same data type as that of RNN data tensor. If null pointer is passed
    /// in, then padding position in output will be undefined.
    //--------------------------------------------------------------------------
    double padding_fill_;
};

} // namespace ManageDescriptor
} // namespace RecurrentNeuralNetwork

#endif // RECURRENT_NEURAL_NETWORK_MANAGE_DESCRIPTOR_SET_DATA_DESCRIPTOR_H