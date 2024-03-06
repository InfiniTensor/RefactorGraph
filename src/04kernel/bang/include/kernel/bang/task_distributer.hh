#ifndef KERNEL_BANG_TASK_DISTRIBUTER_HH
#define KERNEL_BANG_TASK_DISTRIBUTER_HH

#include <cnrt.h>

namespace refactor::kernel::bang {

    /// @brief 内核的启动参数。
    struct KernelLaunchParameters {
        /// @brief 任务规模。
        cnrtDim3_t taskDim;
        /// @brief 任务类型。
        cnrtFunctionType_t funcType;
        /// @brief 用于执行内核的流。
        cnrtQueue_t queue;
    };

}// namespace refactor::kernel::bang

#endif// KERNEL_BANG_TASK_DISTRIBUTER_HH
