#ifndef KERNEL_NVRTC_REPO_H
#define KERNEL_NVRTC_REPO_H

#include "common.h"
#include <cuda.h>

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
        void launch(unsigned int gridDimX,
                    unsigned int gridDimY,
                    unsigned int gridDimZ,
                    unsigned int blockDimX,
                    unsigned int blockDimY,
                    unsigned int blockDimZ,
                    unsigned int sharedMemBytes,
                    void **kernelParams) const;
    };

    std::string_view memCopyType(size_t);
    std::string_view dataType(DataType);

}// namespace refactor::kernel::nvrtc

#endif// KERNEL_NVRTC_REPO_H
