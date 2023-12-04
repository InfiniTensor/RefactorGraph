#ifndef KERNEL_EXTRA_PADDING_CUH
#define KERNEL_EXTRA_PADDING_CUH

#include "common.h"
#include <optional>
#include <thrust/execution_policy.h>
#include <thrust/tabulate.h>

namespace refactor::kernel {

    struct ExtraPadding {
        DataType dt;
        int nc, sohw, sow, h, w, padH, padW;

        static std::optional<ExtraPadding> build(DataType dt, int const *shape, int const *pads);

        size_t workspace() const;

        void const *operator()(void const *src, void *workspace) const;
    };

    template<class T>
    struct ExtraPaddingFunctor {
        ExtraPadding info;
        void const *src;

        __device__ T operator()(size_t i) const noexcept {
            auto h = i / info.sow,
                 w = i % info.sow;
            if (0 < info.padH) {
                if (h < info.padH) {
                    return 0;
                }
                h -= info.padH;
            } else if (h >= info.h) {
                return 0;
            }
            if (0 < info.padW) {
                if (w < info.padW) {
                    return 0;
                }
                w -= info.padW;
            } else if (w >= info.w) {
                return 0;
            }
            return reinterpret_cast<T const *>(src)[i / info.sohw * info.h * info.w + h * info.w + w];
        }
    };

}// namespace refactor::kernel

#endif// KERNEL_EXTRA_PADDING_CUH
