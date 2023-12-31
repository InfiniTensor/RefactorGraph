#include "cuda_kernel.hh"
#include "hardware/functions.h"
#include "kernel/cuda/threads_distributer.cuh"
#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace refactor::kernel {
    using K = DynamicQuantizeLinearCuda;

    template<class T>
    struct QuantizeMinMax {
        T min, max;
    };

    template<class T>
    struct QuantizeMapMinMaxFunctor {
        __device__ __forceinline__ QuantizeMinMax<T>
        operator()(T x) const noexcept {
            return {x, x};
        }
    };

    template<class T>
    struct QuantizeReduceMinMaxFunctor {
        __device__ __forceinline__ QuantizeMinMax<T>
        operator()(QuantizeMinMax<T> a, QuantizeMinMax<T> b) const noexcept {
            return {CUB_MIN(a.min, b.min), CUB_MAX(a.max, b.max)};
        }
    };

    template<class T>
    constexpr static auto
        ZERO = static_cast<T>(0);

    template<class TI, class TO>
    constexpr static auto
        QMIN = static_cast<TI>(std::numeric_limits<TO>::min());

    template<class TI, class TO>
    constexpr static auto
        QMAX = static_cast<TI>(std::numeric_limits<TO>::max());

    template<class TI, class TO>
    constexpr static auto
        QLEN = QMAX<TI, TO> - QMIN<TI, TO>;

    template<class TI, class TO>
    __global__ static void kernel(
        size_t n,
        QuantizeMinMax<TI> const *__restrict__ minmax,
        TI const *__restrict__ x,
        TO *__restrict__ y,
        TI *__restrict__ scale_,
        TO *__restrict__ zp_) {

        auto const [min, max] = *minmax;
        auto cover0 = QuantizeReduceMinMaxFunctor<TI>{}({min, max}, {ZERO<TI>, ZERO<TI>});
        auto scale = (cover0.max - cover0.min) / QLEN<TI, TO>;
        auto zp = static_cast<TO>(round(QMIN<TI, TO> - min / scale));

        auto tid = blockIdx.x * blockDim.x + threadIdx.x;
        for (auto step = blockDim.x * gridDim.x,
                  i = tid;
             i < n;
             i += step) {
            y[i] = static_cast<TO>(std::round(x[i] / scale) + zp);
        }
        if (tid == 0) {
            *scale_ = scale;
            *zp_ = zp;
        }
    }

    auto K::lower(Resources &) const -> RoutineWorkspace {
        using namespace runtime;
        using TI = float;
        using TO = uint8_t;

        constexpr static auto
            _MIN = std::numeric_limits<TI>::min(),
            _MAX = std::numeric_limits<TI>::max();

        auto workspaceSize = hardware::alignBytes(size * sizeof(QuantizeMinMax<TI>), 256);

        QuantizeMinMax<TI> *nullTyped = nullptr;
        size_t tempStorageBytes = 0;
        auto e = cub::DeviceReduce::Reduce(
            nullptr, tempStorageBytes,
            nullTyped, nullTyped, size,
            QuantizeReduceMinMaxFunctor<TI>{},
            QuantizeMinMax<TI>{});
        if (e != cudaSuccess) {
            RUNTIME_ERROR(fmt::format("error: {} {}", (int) e, cudaGetErrorString(e)));
        }

        auto offset0 = workspaceSize;
        workspaceSize += tempStorageBytes;
        workspaceSize = hardware::alignBytes(workspaceSize, 256);

        auto offset1 = workspaceSize;
        workspaceSize += sizeof(QuantizeMinMax<TI>);

        auto routine = [offset0, tempStorageBytes, offset1,
                        params = cuda::ThreadsDistributer()(size)]//
            (Resources &, void *workspacePtr, void const *const *inputs, void *const *outputs) {
                auto x = reinterpret_cast<TI const *>(inputs[0]);
                auto y = reinterpret_cast<TO *>(outputs[0]);
                auto scale = reinterpret_cast<TI *>(outputs[1]);
                auto zp = reinterpret_cast<TO *>(outputs[2]);
                auto workspace = reinterpret_cast<uint8_t *>(workspacePtr);
                auto doubled = reinterpret_cast<QuantizeMinMax<TI> *>(workspace);
                auto tempStorage = workspace + offset0;
                auto minmax = reinterpret_cast<QuantizeMinMax<TI> *>(workspace + offset1);

                thrust::transform(
                    thrust::device,
                    x, x + params.n, doubled,
                    QuantizeMapMinMaxFunctor<TI>{});

                auto tempStorageSize_ = tempStorageBytes;
                auto e = cub::DeviceReduce::Reduce(
                    tempStorage, tempStorageSize_,
                    doubled, minmax, params.n,
                    QuantizeReduceMinMaxFunctor<TI>{},
                    QuantizeMinMax<TI>{_MAX, _MIN});
                if (e != cudaSuccess) {
                    RUNTIME_ERROR(fmt::format("error: {} {}", (int) e, cudaGetErrorString(e)));
                }

                kernel<<<params.gridSize, params.blockSize>>>(
                    params.n, minmax, x, y, scale, zp);
            };

        return {std::move(routine), workspaceSize};
    }

}// namespace refactor::kernel
