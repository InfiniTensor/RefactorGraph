#include "cpu_kernel.hh"
#include "../mat_mul_common/cpu_template.hpp"
#include <execution>

namespace refactor::kernel {
    using K = MatMulIntegerCpu;
    using DT = DataType;

    K::MatMulIntegerCpu(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(decltype(info) info) noexcept -> KernelBox {
        return std::make_unique<K>(std::move(info));
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing MatMulInteger using CPU";
    }

    template<class T> static int8_t sub(T, T);
    template<> int8_t sub<int8_t>(int8_t a, int8_t b) { return a - b; }
    template<> int8_t sub<uint8_t>(uint8_t a, uint8_t b) { return static_cast<int8_t>(static_cast<int16_t>(a) - static_cast<int16_t>(b)); }

    template<class T>
    static void applyZeroPointScalar(
        size_t size, int8_t *dst, void const *src_, void const *zp_) {

        auto src = reinterpret_cast<T const *>(src_);
        auto zp = *reinterpret_cast<T const *>(zp_);
        std::transform(std::execution::par_unseq,
                       src, src + size,
                       dst, [zp](auto x) { return sub(x, zp); });
    }
    template<class T>
    static void applyZeroPointA(
        dim_t b, dim_t m, dim_t n,
        int8_t *dst, void const *src_, void const *zp_) {

        auto src = reinterpret_cast<T const *>(src_),
             zp = reinterpret_cast<T const *>(zp_);
        for (auto i : range0_(b))
            for (auto j : range0_(m))
                for (auto k : range0_(n))
                    dst[i * m * n + j * n + k] = sub(src[i * m * n + j * n + k], zp[i * m + j]);
    }
    template<class T>
    static void applyZeroPointB(
        dim_t b, dim_t m, dim_t n,
        int8_t *dst, void const *src_, void const *zp_) {

        auto src = reinterpret_cast<T const *>(src_),
             zp = reinterpret_cast<T const *>(zp_);
        for (auto i : range0_(b))
            for (auto j : range0_(m))
                for (auto k : range0_(n))
                    dst[i * m * n + j * n + k] = sub(src[i * m * n + j * n + k], zp[i * n + k]);
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        using namespace runtime;

        size_t workspace = 0;
        if (info.a.withZeroPoint) {
            workspace += info.batch() * info.m * info.k;
        }
        if (info.b.withZeroPoint) {
            workspace += info.batch() * info.k * info.n;
        }

        auto routine = [info = info](Resources &, void *workspace, void const *const *inputs, void *const *outputs) {
            auto workspacePtr = reinterpret_cast<int8_t *>(workspace);
            auto a = reinterpret_cast<int8_t const *>(inputs[0]),
                 b = reinterpret_cast<int8_t const *>(inputs[1]);
            auto y = reinterpret_cast<int32_t *>(outputs[0]);

            if (auto meta = info.a; meta.withZeroPoint) {
                auto size = info.batch() * info.m * info.k;
                auto zp = inputs[2];
                if (meta.scalar) {
                    if (meta.signed_) {
                        applyZeroPointScalar<int8_t>(size, workspacePtr, a, zp);
                    } else {
                        applyZeroPointScalar<uint8_t>(size, workspacePtr, a, zp);
                    }
                } else {
                    if (meta.signed_) {
                        applyZeroPointA<int8_t>(info.batch(), info.m, info.k, workspacePtr, a, zp);
                    } else {
                        applyZeroPointA<uint8_t>(info.batch(), info.m, info.k, workspacePtr, a, zp);
                    }
                }
                a = workspacePtr;
                workspacePtr += size;
            }
            if (auto meta = info.b; meta.withZeroPoint) {
                auto size = info.batch() * info.k * info.n;
                auto zp = inputs[3];
                if (meta.scalar) {
                    if (meta.signed_) {
                        applyZeroPointScalar<int8_t>(size, workspacePtr, b, zp);
                    } else {
                        applyZeroPointScalar<uint8_t>(size, workspacePtr, b, zp);
                    }
                } else {
                    if (meta.signed_) {
                        applyZeroPointA<int8_t>(info.batch(), info.k, info.n, workspacePtr, b, zp);
                    } else {
                        applyZeroPointA<uint8_t>(info.batch(), info.k, info.n, workspacePtr, b, zp);
                    }
                }
                b = workspacePtr;
            }

            MatMulCPUMetaData<int32_t, int8_t> const md{
                .M = info.m,
                .K = info.k,
                .N = info.n,
                .strideA0 = info.k,
                .strideA1 = 1,
                .strideB0 = info.n,
                .strideB1 = 1,
                .alpha = 1,
                .beta = 0,
            };
            auto const stepY = info.m * info.n,
                       stepA = info.m * info.k,
                       stepB = info.k * info.n;

            if (info.broadcaster.needBroadcast()) {
                dim_t offset[2];
                for (auto i : range0_(info.batch())) {
                    info.broadcaster.locate(i, offset);
                    md.matrixMultiply(a + stepA * offset[0], b + stepB * offset[1], y + stepY * i);
                }
            } else {
                for (auto i : range0_(info.batch())) {
                    md.matrixMultiply(a + stepA * i, b + stepB * i, y + stepY * i);
                }
            }
        };

        return {std::move(routine), workspace};
    };

}// namespace refactor::kernel
