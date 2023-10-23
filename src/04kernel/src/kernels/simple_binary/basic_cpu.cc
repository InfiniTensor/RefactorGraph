#include "basic_cpu.hh"
#include <execution>
#include <unordered_set>

namespace refactor::kernel {
    using K = BinaryBasicCpu;
    using Op = SimpleBinaryType;
    using DT = DataType;

    constexpr static BinaryBroadcast::Dimension
        A{0, true, false},
        B{0, false, true},
        AB{0, true, true};

    auto BinaryBroadcast::push(
        Dimension next,
        Dimension state,
        uint32_t size) noexcept -> Dimension {
        //          2  3  5
        // 1  2  1  1  3  1
        // 1↓ 2↓ 1↓ 2↑ 3- 5↑
        if (state.code == next.code) {
            dims.back().size *= size;
        } else {
            dims.back().code;
            dims.push_back({size, next.a, next.b});
        }
        return next;
    }

    BinaryBroadcast::BinaryBroadcast(Shape const &a_, Shape const &b_) noexcept
        : dims() {
        Dimension state{0, false, false};
        auto ita = a_.rbegin();
        auto itb = b_.rbegin();
        while (true) {
            if (itb == b_.rend()) {
                if (ita == a_.rend()) { break; }
                // a is longer
                push(A, state, std::accumulate(ita, a_.rend(), 1, std::multiplies<>()));
                break;
            }
            if (ita == a_.rend()) {
                // b is longer
                push(B, state, std::accumulate(itb, b_.rend(), 1, std::multiplies<>()));
                break;
            }
            auto a = *ita++, b = *itb++;
            if (b == 1) {
                if (a == 1) { continue; }
                // broadcast to a
                state = push(A, state, a);
            } else if (a == 1) {
                // broadcast to b
                state = push(B, state, b);
            } else {
                ASSERT(a == b, "a and b must be equal");
                state = push(AB, state, a);
            }
        }
        std::reverse(dims.begin(), dims.end());
    }

    auto BinaryBroadcast::size() const noexcept -> uint32_t {
        return dims.empty() ? 1
                            : std::accumulate(
                                  dims.begin(), dims.end(), 1,
                                  [](auto acc, auto it) { return acc * it.size; });
    }

    auto BinaryBroadcast::locate(uint32_t k) const noexcept
        -> std::pair<uint32_t, uint32_t> {
        std::vector<uint32_t> strides(dims.size() * 3, 1);
        for (uint32_t i = dims.size() - 1; i > 0; --i) {
            strides[3 * (i - 1) + 0] = strides[3 * i + 0] * dims[i].size;
            strides[3 * (i - 1) + 1] = strides[3 * i + 1] * (dims[i].a ? dims[i].size : 1);
            strides[3 * (i - 1) + 2] = strides[3 * i + 2] * (dims[i].b ? dims[i].size : 1);
        }

        uint32_t a = 0, b = 0;
        for (auto i : range0_(dims.size())) {
            auto d = std::div(static_cast<long>(k), static_cast<long>(strides[3 * i]));
            k = d.rem;
            a += strides[3 * i + 1] * d.quot;
            b += strides[3 * i + 2] * d.quot;
        }
        return {a, b};
    }

    K::BinaryBasicCpu(Op opType_, DT dataType_, BinaryBroadcast b) noexcept
        : Kernel(),
          dataType(dataType_),
          opType(opType_),
          broadcast(std::move(b)) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b) noexcept -> KernelBox {
        if (op == Op::Pow) { return nullptr; }// TODO: 暂时不支持
        return a.dataType.isCpuNumberic()
                   ? std::make_unique<K>(op, a.dataType, BinaryBroadcast(a.shape, b.shape))
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
        return "Performing simple operation of 2 tensors on generic cpu";
    }

#define CASE_DT(OP, T)                                                                                    \
    case DT::T:                                                                                           \
        return [broadcast = this->broadcast](runtime::Resources &, void const **inputs, void **outputs) { \
            using T_ = primitive_t<DT::T>::type;                                                          \
            auto a = static_cast<T_ const *>(inputs[0]);                                                  \
            auto b = static_cast<T_ const *>(inputs[1]);                                                  \
            auto c = static_cast<T_ *>(outputs[0]);                                                       \
            for (auto i : range0_(broadcast.size())) {                                                    \
                auto [ia, ib] = broadcast.locate(i);                                                      \
                c[i] = OP(a[ia], b[ib]);                                                                  \
            }                                                                                             \
        }

#define CASE_OP(NAME, LAMBDA)        \
    case Op::NAME:                   \
        switch (dataType.internal) { \
            CASE_DT(LAMBDA, F32);    \
            CASE_DT(LAMBDA, U8);     \
            CASE_DT(LAMBDA, I8);     \
            CASE_DT(LAMBDA, U16);    \
            CASE_DT(LAMBDA, I16);    \
            CASE_DT(LAMBDA, I32);    \
            CASE_DT(LAMBDA, I64);    \
            CASE_DT(LAMBDA, F64);    \
            CASE_DT(LAMBDA, U32);    \
            CASE_DT(LAMBDA, U64);    \
            default:                 \
                UNREACHABLE();       \
        }

    auto K::lower() const noexcept -> Routine {
        using namespace runtime;

        switch (opType) {
            CASE_OP(Add, [](auto a, auto b) { return a + b; })
            CASE_OP(Sub, [](auto a, auto b) { return a - b; })
            CASE_OP(Mul, [](auto a, auto b) { return a * b; })
            CASE_OP(Div, [](auto a, auto b) { return a / b; })
            case Op::And:
                switch (dataType.internal) {
                    CASE_DT([](auto a, auto b) { return a && b; }, Bool);
                    default:
                        UNREACHABLE();
                }
            case Op::Or:
                switch (dataType.internal) {
                    CASE_DT([](auto a, auto b) { return a || b; }, Bool);
                    default:
                        UNREACHABLE();
                }
            case Op::Xor:
                switch (dataType.internal) {
                    CASE_DT([](auto a, auto b) { return a ^ b; }, Bool);
                    default:
                        UNREACHABLE();
                }
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
