#ifndef TRANSFORMER_MANAGE_DESCRIPTOR_SET_ATTENTION_DESCRIPTOR_H
#define TRANSFORMER_MANAGE_DESCRIPTOR_SET_ATTENTION_DESCRIPTOR_H

#include "AttentionDescriptor.h"
#include "RecurrentNeuralNetwork/ManageDescriptor/DropoutDescriptor.h"
#include "Transformer/Attention/Parameters.h"
#include "Utilities/ErrorHandling/HandleUnsuccessfulCuDNNCall.h"

#include <cudnn.h>

namespace Transformer
{
namespace ManageDescriptor
{

//------------------------------------------------------------------------------
/// \param [out] parameters
/// \param [in] attention_dropout_descriptor = Descriptor of dropout
/// operation applied to softmax output.
/// \param [out] post_dropout_descriptor - Descriptor of dropout operation
/// applied to multi-head attention output, just before point where residual
/// connections added.
//------------------------------------------------------------------------------
Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall set_attention_descriptor(
  AttentionDescriptor& descriptor,
  const Transformer::Attention::Parameters& parameters,
  RecurrentNeuralNetwork::ManageDescriptor::DropoutDescriptor&
    attention_dropout_descriptor,
  RecurrentNeuralNetwork::ManageDescriptor::DropoutDescriptor&
    post_dropout_descriptor);

//------------------------------------------------------------------------------
/// \param [in] attnDesc or i.e. descriptor - Attention descriptor
/// \param [out] parameters
/// \param [out] attnDropoutDesc or i.e. attention_dropout_descriptor -
/// Descriptor of the dropout operation applied to softmax output.
/// \param [out] postDropoutDesc
//------------------------------------------------------------------------------
Utilities::ErrorHandling::HandleUnsuccessfulCuDNNCall get_attention_descriptor(
  AttentionDescriptor& descriptor,
  Transformer::Attention::Parameters& parameters,
  RecurrentNeuralNetwork::ManageDescriptor::DropoutDescriptor&
    attention_dropout_descriptor,
  RecurrentNeuralNetwork::ManageDescriptor::DropoutDescriptor&
    post_dropout_descriptor);

} // namespace ManageDescriptor
} // namespace Transformer

#endif // TRANSFORMER_MANAGE_DESCRIPTOR_SET_ATTENTION_DESCRIPTOR_H