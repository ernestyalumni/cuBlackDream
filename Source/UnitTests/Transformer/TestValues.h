#ifndef UNIT_TESTS_TRANSFORMER_TEST_VALUES_H
#define UNIT_TESTS_TRANSFORMER_TEST_VALUES_H

#include "Transformer/Attention/Parameters.h"

namespace GoogleUnitTests
{
namespace Transformer
{

struct ExampleParameters : public ::Transformer::Attention::Parameters
{
  using Parameters::Parameters;
  ExampleParameters();
};

} // namespace Transformer
} // namespace GoogleUnitTests

#endif // UNIT_TESTS_TRANSFORMER_TEST_VALUES_H