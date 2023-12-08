#ifndef KERNEL_NVRTC_REPO_H
#define KERNEL_NVRTC_REPO_H

#include "common.h"
#include <cuda.h>

#define CUDA_ASSERT(CALL)                                                             \
    if (auto result = CALL; result != CUDA_SUCCESS) {                                 \
        const char *msg;                                                              \
        cuGetErrorName(result, &msg);                                                 \
        RUNTIME_ERROR(fmt::format("cuda driver failed on \"" #CALL "\" with {} ({})", \
                                  msg, (int) result));                                \
    }

namespace refactor::kernel::nvrtc {

    class Handler {
        CUmodule _module;
        CUfunction _kernel;

        Handler(std::string_view name,
                std::string_view code,
                std::string_view symbol);

    public:
        ~Handler();

        static Arc<Handler> compile(
            std::string_view name,
            std::string_view code,
            std::string_view symbol);
        CUfunction kernel() const;
    };

    std::string_view memCopyType(size_t);
    std::string_view dataType(DataType);

}// namespace refactor::kernel::nvrtc

#endif// KERNEL_NVRTC_REPO_H
