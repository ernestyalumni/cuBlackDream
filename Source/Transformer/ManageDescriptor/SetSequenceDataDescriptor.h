#ifndef TRANSFORMER_MANAGE_DESCRIPTOR_SET_ATTENTION_DESCRIPTOR_H
#define TRANSFORMER_MANAGE_DESCRIPTOR_SET_ATTENTION_DESCRIPTOR_H

#include "SequenceDataDescriptor.h"
#include "Transformer/Attention/Parameters.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <array>
#include <cstddef>
#include <cudnn.h>
#include <vector>

namespace Transformer
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
/// \ref 7.2.53. cudnnSetSeqDataDescriptor()
/// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetSeqDataDescriptor
/// cudnnStatus_t cudnnSetSeqDataDescriptor(
///   cudnnSeqDataDescriptor_t seqDataDesc,
///   cudnnDataType_t dataType,
///   int nbDims,
///   const int dimA[],
///   const cudnnSeqDataAxis_t axes[],
///   size_t seqLengthArraySize
///   const int seqLengthArray[],
///   void* paddingFill);
///
/// Different sequences are bundled together in a batch. A BATCH may be a group
/// of individual sequences or beams. A beam is a cluster of alternative
/// sequences or candidates.
///
/// On paddingFill: Every sequence can have a different length, even within same
/// beam, so vectors towards end of sequence can be just padding. The
/// paddingFill argument specifies how padding vector should be written in
/// output sequence data buffers. paddingFill argument points to 1 value of type
/// dataType taht shoudl be copied to all elements in padding vectors.
/// Currently, only supported value for paddingFill is NULL which means this
/// option should be ignored.
///
/// The seqLengthArray[] must specify all sequence lengths in container so total
/// size of this array should be dimA[CUDNN_SEQDATA_BATCH_DIM] *
/// dimA[CUDNN_SEQDATA_BEAM_DIM]
///
/// \param [out] seqDataDesc
/// \param [in] dimA[] - Integer array specifying sequence data dimensions. Use
/// cudnnSeqDataAxis_t enumerated type to index all active dimA[] elements.
/// \param [in] axes[] - Array of cudnnSeqDataAxis_t that defines layout of
/// sequence data in memory. First nbDims elements of axes[] should be
/// initialized with outermost dimension in axes[0] and innermost dimension in
/// axes[nbDims-1]
/// \param [in] seqLengthArray[] - An integer array that defines all sequence
/// lengths of the container.
/// \param [in] paddingFill - must be NULL Pointer to a value of dataType that's
/// used to fill up output vectors beyond valid length of each sequence or NULL
/// to ignore this setting.
//------------------------------------------------------------------------------

class SetSequenceDataDescriptor
{
  public:
  
    static constexpr int EXPECTED_NUMBER_OF_ACTIVE_DIMENSIONS_ {4};

    class CreateDimensionsAndSequenceLengths
    {
      public:

        CreateDimensionsAndSequenceLengths() = delete;

        CreateDimensionsAndSequenceLengths(
          const int time_dimension,
          const int batch_dimension,
          const int beam_dimension,
          const int vector_dimension);

        //----------------------------------------------------------------------
        /// \ref 7.2.53. cudnnSetSeqDataDescriptor()
        /// Each element of seqLengthArray[] array should have a non-negative
        /// value, less than or equal to dimA[CUDNN_SEQDATA_TIME_DIM], the
        /// maximum sequence length.
        //----------------------------------------------------------------------
        void set_sequence_length_array(
          const std::vector<int>& input_sequence_lengths);

        void fill_with_maximum_sequence_length();

      private:

        std::array<int, 4> dimensions_array_;

        std::vector<int> sequence_length_array_;
        //----------------------------------------------------------------------
        /// seqLengthArray[] - must specify all sequence lenghts in container,
        /// so total size of this array should be
        /// dimA[CUDNN_SEQDATA_BATCH_DIM] * dimA[CUDNN_SEQDATA_BEAM_DIM].
        //----------------------------------------------------------------------
        std::size_t sequence_length_array_total_size_;
    };

    SetSequenceDataDescriptor();

    Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall set_descriptor(
      SequenceDataDescriptor& descriptor);

  private:

    //--------------------------------------------------------------------------
    /// Integer array specifying sequence data dimensions.
    //--------------------------------------------------------------------------
    int* dimensions_array_;

    cudnnSeqDataAxis_t* axes_;

    std::size_t* sequence_length_array_;
};

} // namespace ManageDescriptor
} // namespace Transformer

#endif // TRANSFORMER_MANAGE_DESCRIPTOR_SET_SEQUENCE_DATA_DESCRIPTOR_H