#include "cuda_kernel.hh"

#ifdef USE_CUDA
#include "../../generator/nvrtc_repo.h"
#include "kernel/cuda/threads_distributer.cuh"
#endif

namespace refactor::kernel {
    using K = BinaryCuda;
    using Op = SimpleBinaryType;
    using DT = DataType;

    K::BinaryCuda(Op opType_, DT dataType_, Broadcaster b) noexcept
        : Kernel(),
          dataType(dataType_),
          opType(opType_),
          broadcaster(std::move(b)) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b) noexcept -> KernelBox {
#ifndef USE_CUDA
        return nullptr;
#endif

        return a.dataType == b.dataType
                   ? std::make_unique<K>(op, a.dataType, Broadcaster({a, b}))
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
        return "Performing binary operation of 2 tensors on Nvidai GPU";
    }

#ifdef USE_CUDA

    constexpr static const char *NO_BROADCAST = R"~(
__device__ __forceinline__ static {0:} fn({0:} a, {0:} b) {{
    return {1:};
}}

extern "C" __global__ void kernel(
    {0:}       *__restrict__ c,
    {0:} const *__restrict__ a,
    {0:} const *__restrict__ b,
    size_t n
) {{
    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < n;
         tid += step) {{
        c[tid] = fn(a[tid], b[tid]);
    }}
}}
)~";

    constexpr static const char *SCALAR = R"~(
__device__ __forceinline__ static {0:} fn({0:} a, {0:} b) {{
    return {1:};
}}

extern "C" __global__ void kernel(
    {0:}       *__restrict__ y,
    {0:} const *__restrict__ v,
    {0:} const *__restrict__ s,
    size_t n
) {{
    auto num = *s;
    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < n;
         tid += step) {{
        y[tid] = fn(v[tid], num);
    }}
}}
)~";

    constexpr static const char *BROADCAST = R"~(
struct Strides {{
    unsigned int s[3 * {2:}];
}};

__device__ __forceinline__ static {0:} fn({0:} a, {0:} b) {{
    return {1:};
}}

extern "C" __global__ void kernel(
    {0:}       *__restrict__ c,
    {0:} const *__restrict__ a,
    {0:} const *__restrict__ b,
    Strides strides,
    size_t n
) {{
    for (auto tid = blockIdx.x * blockDim.x + threadIdx.x,
              step = blockDim.x * gridDim.x;
         tid < n;
         tid += step) {{
        unsigned int ia = 0, ib = 0;
        auto rem = tid;
        for (auto i = 0; i < {2:}; ++i) {{
            auto dim = strides.s + 3 * i;
            auto quot = rem / dim[2];
            ia += dim[0] * quot;
            ib += dim[1] * quot;
            rem %= dim[2];
        }}
        c[tid] = fn(a[ia], b[ib]);
    }}
}}
)~";

    constexpr static std::string_view op(SimpleBinaryType op, DataType dt) {
        switch (op) {
            case SimpleBinaryType::Add:
                return "a + b";
            case SimpleBinaryType::Sub:
                return "a - b";
            case SimpleBinaryType::Mul:
                return "a * b";
            case SimpleBinaryType::Div:
                return "a / b";
            case SimpleBinaryType::And:
                return "a && b";
            case SimpleBinaryType::Or:
                return "a || b";
            case SimpleBinaryType::Xor:
                return "a ^ b";
            case SimpleBinaryType::Pow:
                switch (dt) {
                    case DataType::F32:
                        return "powf(a, b)";
                    case DataType::FP16:
                        return "__float2half(powf(__half2float(a), __half2float(b)))";
                    case DataType::BF16:
                        return "__float2bfloat16(powf(__bfloat162float(a), __bfloat162float(b)))";
                    default:
                        return "pow(a, b)";
                }
            case SimpleBinaryType::Mod:
                switch (dt) {
                    case DataType::U8:
                    case DataType::I8:
                    case DataType::U16:
                    case DataType::I16:
                    case DataType::I32:
                    case DataType::I64:
                    case DataType::U32:
                    case DataType::U64:
                        return "a % b";
                    default:
                        UNREACHABLE();
                }
            case SimpleBinaryType::Fmod:
                switch (dt) {
                    case DataType::U8:
                    case DataType::U16:
                    case DataType::U32:
                    case DataType::U64:
                        return "a % b";
                    case DataType::I8:
                        return "static_cast<char>(fmodf(a, b))";
                    case DataType::I16:
                        return "static_cast<short>(fmodf(a, b))";
                    case DataType::I32:
                        return "static_cast<int>(fmodf(a, b))";
                    case DataType::I64:
                        return "static_cast<long long>(fmodf(a, b))";
                    case DataType::F32:
                        return "fmodf(a, b)";
                    case DataType::FP16:
                        return "__float2half(fmodf(__half2float(a), __half2float(b)))";
                    case DataType::BF16:
                        return "__float2bfloat16(fmodf(__bfloat162float(a), __bfloat162float(b)))";
                    default:
                        UNREACHABLE();
                }
            default:
                UNREACHABLE();
        }
    }

    auto K::lower(Resources &res) const -> RoutineWorkspace {
        using namespace runtime;

        auto postfix = fmt::format("_{}_{}", dataType.name(), opName(opType));
        auto dt_ = nvrtc::dataType(dataType);
        auto op_ = op(opType, dataType);
        auto params = cuda::ThreadsDistributer()(broadcaster.outputsCount);

        if (!broadcaster.needBroadcast()) {
            auto name = fmt::format("binary{}", postfix);
            auto code = fmt::format(NO_BROADCAST, dt_, op_);
            return [params, h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel")]//
                (Resources &, void *, void const *const *inputs, void *const *outputs) {
                    auto c = outputs[0];
                    auto a = inputs[0],
                         b = inputs[1];
                    auto n = params.n;
                    void *args[]{&c, &a, &b, &n};
                    h->launch(params.gridSize, 1, 1,
                              params.blockSize, 1, 1,
                              0, args);
                };

        } else if (auto rank = broadcaster.strides.size() / (broadcaster.inputsCount + 1); rank == 1) {
            static const std::vector<dim_t> S0{0, 1, 1}, S1{1, 0, 1};
            auto name = fmt::format("binaryScalar{}", postfix);
            auto code = fmt::format(SCALAR, dt_, op_);
            return [params, h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel"),
                    // clang-format off
                    scalar = broadcaster.strides == S0 ? 0
                           : broadcaster.strides == S1 ? 1
                           : UNREACHABLEX(int, "Unreachable")]// clang-format on
                (Resources &, void *, void const *const *inputs, void *const *outputs) {
                    auto c = outputs[0];
                    auto s = inputs[scalar],
                         v = inputs[1 - scalar];
                    auto n = params.n;
                    void *args[]{&c, &v, &s, &n};
                    h->launch(params.gridSize, 1, 1,
                              params.blockSize, 1, 1,
                              0, args);
                };

        } else {
            auto name = fmt::format("binary{}{}", rank, postfix);
            auto code = fmt::format(BROADCAST, dt_, op_, rank);
            return [params, h = nvrtc::Handler::compile(name.c_str(), code.c_str(), "kernel"),
                    strides = broadcaster.strides]//
                (Resources &, void *, void const *const *inputs, void *const *outputs) {
                    auto c = outputs[0];
                    auto a = inputs[0],
                         b = inputs[1];
                    auto n = params.n;
                    void *args[]{&c, &a, &b, const_cast<dim_t *>(strides.data()), &n};
                    h->launch(params.gridSize, 1, 1,
                              params.blockSize, 1, 1,
                              0, args);
                };
        }
    }

#endif

}// namespace refactor::kernel
