#include "cuda_kernel.hh"
#include <unordered_set>

#ifdef USE_CUDA
#include "../../generator/nvrtc_repo.h"
#include "kernel/cuda/threads_distributer.cuh"
#include <cuda_runtime.h>
#endif

namespace refactor::kernel {
    using K = SimpleUnaryCuda;
    using Op = SimpleUnaryType;
    using DT = DataType;

    K::SimpleUnaryCuda(Op opType_, DT dataType_, size_t size_) noexcept
        : Kernel(), dataType(dataType_), opType(opType_), size(size_) {}

    auto K::build(Op op, Tensor const &a) noexcept -> KernelBox {
        static const std::unordered_set<Op>
            supportedOp{Op::Abs, Op::Relu, Op::Sqrt,
                        Op::Sigmoid, Op::Tanh, Op::Neg,
                        Op::Erf, Op::HardSwish};
#ifndef USE_CUDA
        return nullptr;
#endif

        return supportedOp.contains(op)
                   ? std::make_unique<K>(op, a.dataType, a.elementsSize())
                   : nullptr;
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing unary operation on Nvidia GPU";
    }

#ifdef USE_CUDA

    constexpr static const char *TEMPLATE = R"~(
__device__ __forceinline__ static {0:} fn({0:} x) {{
    return {1:};
}}

extern "C" __global__ void kernel(
    {0:}       *__restrict__ y,
    {0:} const *__restrict__ x,
    size_t n
) {{
    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < n;
         tid += step)
        y[tid] = fn(x[tid]);
}}
)~";

    constexpr static uint16_t __(Op op, DT dt) {
        return (static_cast<uint16_t>(op) << 8) | static_cast<uint16_t>(dt);
    }

    auto K::lower(Resources &res) const -> RoutineWorkspace {
        using namespace runtime;

        if (dataType.isUnsigned()) {
            switch (opType) {
                case Op::Abs:
                case Op::Relu:
                    return [n = size * dataType.size()](Resources &, void *, void const *const *inputs, void *const *outputs) {
                        if (outputs[0] != inputs[0]) {
                            cudaMemcpyAsync(outputs[0], inputs[0], n, cudaMemcpyDeviceToDevice);
                        }
                    };
                case Op::Neg:
                    UNREACHABLE();
                default:
                    break;
            }
        }

        // clang-format off
        // see <https://docs.nvidia.com/cuda/cuda-math-api/index.html>.
        static const std::unordered_map<uint16_t, std::string_view> op {
            {__(Op::Abs, DT::I8  ), "x >= 0 ? x : -x"},
            {__(Op::Abs, DT::I16 ), "x >= 0 ? x : -x"},
            {__(Op::Abs, DT::I32 ), "x >= 0 ? x : -x"},
            {__(Op::Abs, DT::I64 ), "x >= 0 ? x : -x"},
            {__(Op::Abs, DT::FP16), "habs(x)"        },
            {__(Op::Abs, DT::BF16), "habs(x)"        },
            {__(Op::Abs, DT::F32 ), "fabsf(x)"       },
            {__(Op::Abs, DT::F64 ), "fabs(x)"        },

            {__(Op::Relu, DT::I8  ), "x > 0 ? x : 0"},
            {__(Op::Relu, DT::I16 ), "x > 0 ? x : 0"},
            {__(Op::Relu, DT::I32 ), "x > 0 ? x : 0"},
            {__(Op::Relu, DT::I64 ), "x > 0 ? x : 0"},
            {__(Op::Relu, DT::FP16), "x > CUDART_ZERO_FP16 ? x: CUDART_ZERO_FP16"},
            {__(Op::Relu, DT::BF16), "x > CUDART_ZERO_BF16 ? x: CUDART_ZERO_BF16"},
            {__(Op::Relu, DT::F32 ), "x > 0 ? x : 0"},
            {__(Op::Relu, DT::F64 ), "x > 0 ? x : 0"},

            {__(Op::Sqrt, DT::U8  ), "__fsqrt_rn(static_cast<float>(x))" },
            {__(Op::Sqrt, DT::U16 ), "__fsqrt_rn(static_cast<float>(x))" },
            {__(Op::Sqrt, DT::U32 ), "__dsqrt_rn(static_cast<double>(x))" },
            {__(Op::Sqrt, DT::U64 ), "__dsqrt_rn(static_cast<double>(x))" },
            {__(Op::Sqrt, DT::I8  ), "__fsqrt_rn(static_cast<float>(x))" },
            {__(Op::Sqrt, DT::I16 ), "__fsqrt_rn(static_cast<float>(x))" },
            {__(Op::Sqrt, DT::I32 ), "__dsqrt_rn(static_cast<double>(x))" },
            {__(Op::Sqrt, DT::I64 ), "__dsqrt_rn(static_cast<double>(x))" },
            {__(Op::Sqrt, DT::FP16), "hsqrt(x)"                },
            {__(Op::Sqrt, DT::BF16), "hsqrt(x)"                },
            {__(Op::Sqrt, DT::F32 ), "__fsqrt_rn(x)"             },
            {__(Op::Sqrt, DT::F64 ), "__dsqrt_rn(x)"             },

            {__(Op::Sigmoid, DT::U8  ), "fdividef(1, 1 + expf(-static_cast<float>(x)))" },
            {__(Op::Sigmoid, DT::U16 ), "fdividef(1, 1 + expf(-static_cast<float>(x)))" },
            {__(Op::Sigmoid, DT::U32 ),      "1.0 / (1 + exp(-static_cast<double>(x)))" },
            {__(Op::Sigmoid, DT::U64 ),      "1.0 / (1 + exp(-static_cast<double>(x)))" },
            {__(Op::Sigmoid, DT::I8  ), "fdividef(1, 1 + expf(-static_cast<float>(x)))" },
            {__(Op::Sigmoid, DT::I16 ), "fdividef(1, 1 + expf(-static_cast<float>(x)))" },
            {__(Op::Sigmoid, DT::I32 ),      "1.0 / (1 + exp(-static_cast<double>(x)))" },
            {__(Op::Sigmoid, DT::I64 ),      "1.0 / (1 + exp(-static_cast<double>(x)))" },
            {__(Op::Sigmoid, DT::FP16), "hrcp(CUDART_ONE_FP16 + hexp(-x))"              },
            {__(Op::Sigmoid, DT::BF16), "hrcp(CUDART_ONE_BF16 + hexp(-x))"              },
            {__(Op::Sigmoid, DT::F32 ), "fdividef(1, 1 + expf(-x))"                     },
            {__(Op::Sigmoid, DT::F64 ),      "1.0 / (1 + exp(-x))"                      },

            {__(Op::Tanh, DT::F32 ), "tanh(x)"},
            {__(Op::Tanh, DT::F64 ), "tanh(x)"},

            {__(Op::Neg, DT::I8  ), "-x"},
            {__(Op::Neg, DT::I16 ), "-x"},
            {__(Op::Neg, DT::I32 ), "-x"},
            {__(Op::Neg, DT::I64 ), "-x"},
            {__(Op::Neg, DT::FP16), "-x"},
            {__(Op::Neg, DT::BF16), "-x"},
            {__(Op::Neg, DT::F32 ), "-x"},
            {__(Op::Neg, DT::F64 ), "-x"},

            {__(Op::Erf, DT::F32 ), "erff(x)"},
            {__(Op::Erf, DT::F64 ), "erf(x)"},
            {__(Op::Erf, DT::U8  ), "erff(static_cast<float>(x))"},
            {__(Op::Erf, DT::I8  ), "erff(static_cast<float>(x))"},
            {__(Op::Erf, DT::U16 ), "erff(static_cast<float>(x))"},
            {__(Op::Erf, DT::I16 ), "erff(static_cast<float>(x))"},
            {__(Op::Erf, DT::U32 ), "erf(static_cast<double>(x))"},
            {__(Op::Erf, DT::I32 ), "erf(static_cast<double>(x))"},
            {__(Op::Erf, DT::U64 ), "erf(static_cast<double>(x))"},
            {__(Op::Erf, DT::I64 ), "erf(static_cast<double>(x))"},
            {__(Op::Erf, DT::FP16), "__float2half(erff(__half2float(x)))"},
            {__(Op::Erf, DT::BF16), "__float2bfloat16(erff(__bfloat162float(x)))"},

            {__(Op::HardSwish, DT::F32 ), "x * fmaxf(0.f, fminf(1.f, fmaf(1.f/6.f, x, 0.5f)))"},
            {__(Op::HardSwish, DT::FP16), "x * __hmax(CUDART_ZERO_FP16, __hmin(CUDART_ONE_FP16, hrcp(__float2half(6.f)) * x + hrcp(__float2half(2.f))))"},
            {__(Op::HardSwish, DT::F64 ), "x * fmax(0.0, fmin(1.0, fma(1.0/6.0, x, 0.5)))"},

        };
        // clang-format on

        auto name = fmt::format("unary_{}_{}", dataType.name(), unaryName(opType));
        auto code = fmt::format(TEMPLATE, nvrtc::dataType(dataType), op.at(__(opType, dataType)));
        return [h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel"),
                params = cuda::ThreadsDistributer()(size)](
                   Resources &, void *, void const *const *inputs, void *const *outputs) {
            auto y = outputs[0];
            auto x = inputs[0];
            auto n = params.n;
            void *args[]{&y, &x, &n};
            h->launch(params.gridSize, 1, 1,
                      params.blockSize, 1, 1,
                      0, args);
        };
    }

#endif

}// namespace refactor::kernel
