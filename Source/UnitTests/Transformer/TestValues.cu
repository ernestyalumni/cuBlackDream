#include "TestValues.h"

#include <cudnn.h>

namespace GoogleUnitTests
{
namespace Transformer
{

ExampleParameters::ExampleParameters():
  Parameters{
    CUDNN_ATTN_ENABLE_PROJ_BIASES,
    2,
    1.0,
    CUDNN_DATA_FLOAT,
    CUDNN_DATA_FLOAT,
    CUDNN_DEFAULT_MATH,
    512,
    256,
    128,
    // D_q / H = P_q
    64,
    // D_k / H = P_k
    64,
    // D_v / H = P_v
    16,
    8,
    128,
    64,
    32,
    1
    }
{}

} // namespace Transformer
} // namespace GoogleUnitTests