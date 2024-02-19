#ifndef KERNEL_COMMUNICATION_ATTRIBUTES_H
#define KERNEL_COMMUNICATION_ATTRIBUTES_H

namespace refactor::kernel {
    enum class AllReduceType {
        Sum,
        Avg,
        Min,
        Max,
        Prod
    };
}

#endif
