#ifndef ACTIVATION_MANAGE_DESCRIPTOR_ACTIVATION_DESCRIPTOR_H
#define ACTIVATION_MANAGE_DESCRIPTOR_ACTIVATION_DESCRIPTOR_H

#include <cudnn.h>

namespace Activation
{

namespace ManageDescriptor
{

class ActivationDescriptor
{
  public:

    ActivationDescriptor();

    ~ActivationDescriptor();

    //--------------------------------------------------------------------------
    /// https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnActivationDescriptor_t
    /// \ref 3.1.1.1
    /// \details cudnnActivationDescriptor_t is a pointer to an opaque structure
    /// holding the description of an activation operation.
    //--------------------------------------------------------------------------
    cudnnActivationDescriptor_t descriptor_;
};

} // namespace ManageDescriptor
} // namespace Activation

#endif // ACTIVATION_MANAGE_DESCRIPTOR_ACTIVATION_DESCRIPTOR_H