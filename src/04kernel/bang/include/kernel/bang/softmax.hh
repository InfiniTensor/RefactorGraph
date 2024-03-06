#ifndef KERNEL_BANG_SOFTMAX_HH
#define KERNEL_BANG_SOFTMAX_HH

#include "task_distributer.hh"

namespace refactor::kernel::bang {

    template<class T>
    void launchSoftmax(
        KernelLaunchParameters &params,
        void const *input, void *output,
        int &nDim, int &axis,
        int &otherSize, int &frontSize,
        int &dimSize, int &stride);

}// namespace refactor::kernel::bang

#endif// KERNEL_BANG_SOFTMAX_HH
