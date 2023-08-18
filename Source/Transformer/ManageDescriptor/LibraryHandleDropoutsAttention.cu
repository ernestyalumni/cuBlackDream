#include "Transformer/ManageDescriptor/LibraryHandleDropoutsAttention.h"

#include "RecurrentNeuralNetwork/ManageDescriptor/SetDropoutDescriptor.h"
#include "Transformer/Attention/Parameters.h"
#include "Transformer/ManageDescriptor/AttentionDescriptor.h"
#include "Transformer/ManageDescriptor/SetAttentionDescriptor.h"

#include <stdexcept>

using RecurrentNeuralNetwork::ManageDescriptor::SetDropoutDescriptor;
using Transformer::Attention::Parameters;
using Transformer::ManageDescriptor::set_attention_descriptor;

namespace Transformer
{

namespace ManageDescriptor
{

LibraryHandleDropoutsAttention::LibraryHandleDropoutsAttention(
  const Parameters& parameters,
  const float attention_dropout_probability,
  const unsigned long long attention_seed,
  const float post_dropout_probability,
  const unsigned long long post_seed
  ):
  handle_{},
  attention_dropout_descriptor_{},
  post_dropout_descriptor_{},
  descriptor_{},
  attention_dropout_probability_{attention_dropout_probability},
  post_dropout_probability_{post_dropout_probability},
  attention_seed_{attention_seed},
  post_seed_{post_seed}
{
  attention_dropout_descriptor_.get_states_size_for_forward(handle_);
  post_dropout_descriptor_.get_states_size_for_forward(handle_);

  SetDropoutDescriptor set_dropout_descriptor {
    attention_dropout_probability,
    attention_seed};
  auto result = set_dropout_descriptor.set_descriptor(
    attention_dropout_descriptor_,
    handle_);

  if (!result.is_success())
  {
    throw std::runtime_error(result.get_error_message());
  }

  set_dropout_descriptor.dropout_ = post_dropout_probability_;
  set_dropout_descriptor.seed_ = post_seed_;

  result = set_dropout_descriptor.set_descriptor(
    post_dropout_descriptor_,
    handle_);

  if (!result.is_success())
  {
    throw std::runtime_error(result.get_error_message());
  }

  result = set_attention_descriptor(
    descriptor_,
    parameters,
    attention_dropout_descriptor_,
    post_dropout_descriptor_);

  if (!result.is_success())
  {
    throw std::runtime_error(result.get_error_message());
  }
}

} // namespace ManageDescriptor
} // namespace Transformer