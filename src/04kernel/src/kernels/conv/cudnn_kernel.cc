#include "cudnn_kernel.hh"
#include "refactor/common.h"
#include "refactor/common.h"

namespace refactor::kernel {
    using K = ConvCudnn;

    K::ConvCudnn(cudnn::ConvInfo info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(cudnn::ConvolutionFwdAlgo algo,
                  PoolAttributes const &poolAttributes,
                  Tensor const &x,
                  Tensor const &w,
                  Tensor const &y) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        // group is not supported
        if (w.rank() != 4 || x.shape[1] != w.shape[1]) {
            return nullptr;
        }
        auto padsBegin = poolAttributes.padsBegin(),
             padsEnd = poolAttributes.padsEnd();
        if (padsBegin[0] != padsEnd[0] ||
            padsBegin[1] != padsEnd[1]) {
            return nullptr;
        }
        auto d = poolAttributes.dilations(),
             p = poolAttributes.pads(),
             s = poolAttributes.strides();
        return std::make_unique<K>(cudnn::ConvInfo{
            x.dataType,
            algo,
            {
                static_cast<int>(x.shape[0]),
                static_cast<int>(x.shape[1]),
                static_cast<int>(x.shape[2]),
                static_cast<int>(x.shape[3]),
            },
            {
                static_cast<int>(w.shape[0]),
                static_cast<int>(w.shape[1]),
                static_cast<int>(w.shape[2]),
                static_cast<int>(w.shape[3]),
            },
            {
                static_cast<int>(y.shape[0]),
                static_cast<int>(y.shape[1]),
                static_cast<int>(y.shape[2]),
                static_cast<int>(y.shape[3]),
            },
            {d[0], d[1]},
            {p[0], p[1]},
            {s[0], s[1]}});
    }

    auto K::typeId(cudnn::ConvolutionFwdAlgo algo) noexcept -> size_t {
        switch (algo) {
            case cudnn::ConvolutionFwdAlgo::IMPLICIT_GEMM: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case cudnn::ConvolutionFwdAlgo::IMPLICIT_PRECOMP_GEMM: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case cudnn::ConvolutionFwdAlgo::GEMM: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case cudnn::ConvolutionFwdAlgo::DIRECT: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case cudnn::ConvolutionFwdAlgo::FFT: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case cudnn::ConvolutionFwdAlgo::FFT_TILING: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case cudnn::ConvolutionFwdAlgo::WINOGRAD: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case cudnn::ConvolutionFwdAlgo::WINOGRAD_NONFUSED: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            case cudnn::ConvolutionFwdAlgo::COUNT: {
                static uint8_t ID = 1;
                return reinterpret_cast<size_t>(&ID);
            }
            default:
                UNREACHABLE();
        }
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(info.algo); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing conv using CUDNN";
    }
    auto K::lower() const noexcept -> Routine {
        return info.lower();
    }

}// namespace refactor::kernel
