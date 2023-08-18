#ifndef TRANSFORMER_MANAGE_DESCRIPTOR_SEQUENCE_DATA_DESCRIPTOR_H
#define TRANSFORMER_MANAGE_DESCRIPTOR_SEQUENCE_DATA_DESCRIPTOR_H

#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

namespace Transformer
{
namespace ManageDescriptor
{

class SequenceDataDescriptor
{
  public:

    using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    SequenceDataDescriptor();

    ~SequenceDataDescriptor();

    //--------------------------------------------------------------------------
    /// \ref 7.2.7. cudnnCreateSeqDataDescriptor()
    //--------------------------------------------------------------------------
    cudnnSeqDataDescriptor_t descriptor_;
};

} // namespace ManageDescriptor
} // namespace Transformer

#endif // TRANSFORMER_MANAGE_DESCRIPTOR_SEQUENCE_DATA_DESCRIPTOR_H