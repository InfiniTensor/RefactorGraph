﻿#include "cuda_kernel.hh"

namespace refactor::kernel {
    using namespace runtime;

    auto TransposeCuda::lower() const noexcept -> Routine {
        return [](Resources &res, void const **inputs, void **outputs) {
        };
    }

}// namespace refactor::kernel