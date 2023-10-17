#include "kernel/layout.h"
#include "refactor/common.h"

namespace refactor::kernel {

    auto LayoutType::permTo(LayoutType rhs) const noexcept
        -> std::array<uint8_t, 4> {
#define MERGE(FROM, TO) (static_cast<int>(FROM) << 8) | static_cast<int>(TO)
        switch (MERGE(internal, rhs)) {
            case MERGE(NCHW, NHWC):
                return {0, 2, 3, 1};
            case MERGE(NHWC, NCHW):
                return {0, 3, 1, 2};
            case MERGE(NCHW, NCHW):
            case MERGE(NHWC, NHWC):
                return {0, 1, 2, 3};
            default:
                UNREACHABLE();
        }
#undef MERGE
    }

}// namespace refactor::kernel
