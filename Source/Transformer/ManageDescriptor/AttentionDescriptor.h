#ifndef TRANSFORMER_MANAGE_DESCRIPTOR_ATTENTION_DESCRIPTOR_H
#define TRANSFORMER_MANAGE_DESCRIPTOR_ATTENTION_DESCRIPTOR_H

#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

namespace Transformer
{
namespace ManageDescriptor
{

class AttentionDescriptor
{
  public:

    using HandleUnsuccessfulCuDNNCall =
      Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall;

    AttentionDescriptor();

    ~AttentionDescriptor();

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateRNNDescriptor
    /// \ref 7.2.6.
    /// \details cudnnRNNDescriptor_t is a pointer to an opaque structure
    /// holding the description of an RNN operation.
    //--------------------------------------------------------------------------
    cudnnAttnDescriptor_t descriptor_;
};

} // namespace ManageDescriptor
} // namespace Transformer

#endif // TRANSFORMER_MANAGE_DESCRIPTOR_ATTENTION_DESCRIPTOR_H