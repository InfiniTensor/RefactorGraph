#include "kernel/kernel.h"

namespace refactor::kernel {

    RoutineWorkspace Kernel::lower(Resources &) const {
        RUNTIME_ERROR(fmt::format("lower not implemented for {}", description()));
    }

}// namespace refactor::kernel
