#include "basic_cpu.hh"
#include <execution>
#include <unordered_set>

namespace refactor::kernel {
    using K = BinaryBasicCpu;
    using Op = SimpleBinaryType;
    using DT = DataType;

    BinaryBroadcast::BinaryBroadcast(Shape const &a_, Shape const &b_) noexcept
        : _strides(), _size(1) {

        union Dimension {
            struct {
                uint32_t
                    size : 30;
                bool
                    a : 1,
                    b : 1;
            };
            struct {
                uint32_t : 30,
                    state : 2;
            };
        };
        constexpr static Dimension
            A{0, true, false},
            B{0, false, true},
            AB{0, true, true};

        // 折叠同类、相邻的维度
        //          2  3  5
        // 1  2  1  1  3  1
        // 1↓ 2↓ 1↓ 2↑ 3- 5↑
        std::vector<Dimension> dims{{0, false, false}};
        auto push = [&](Dimension next, uint32_t size) {
            if (dims.back().state == next.state) {
                dims.back().size *= size;
            } else {
                dims.push_back({size, next.a, next.b});
            }
        };

        auto ita = a_.rbegin();
        auto itb = b_.rbegin();
        while (true) {
            if (itb == b_.rend()) {
                if (ita == a_.rend()) { break; }
                // a is longer
                push(A, std::accumulate(ita, a_.rend(), 1, std::multiplies<>()));
                break;
            }
            if (ita == a_.rend()) {
                // b is longer
                push(B, std::accumulate(itb, b_.rend(), 1, std::multiplies<>()));
                break;
            }
            auto a = *ita++, b = *itb++;
            if (b == 1) {
                if (a == 1) { continue; }
                // broadcast to a
                push(A, a);
            } else if (a == 1) {
                // broadcast to b
                push(B, b);
            } else {
                ASSERT(a == b, "a and b must be equal");
                push(AB, a);
            }
        }
        if (dims.empty()) {
            return;
        }

        std::reverse(dims.begin(), dims.end());
        auto rank = dims.size() - 1;
        _strides.resize(rank * 3, 1);
        uint_lv2 aMul = 1, bMul = 1, cMul = 1;
        for (auto i : range0_(rank).rev()) {
            _strides[3 * i + 0] = cMul;
            _strides[3 * i + 1] = dims[i].a ? aMul : 0;
            _strides[3 * i + 2] = dims[i].b ? bMul : 0;
            auto size = dims[i].size;
            cMul *= size;
            if (dims[i + 1].a) { aMul *= size; }
            if (dims[i + 1].b) { bMul *= size; }
        }
        _size = dims.empty() ? 1 : _strides[0] * dims[0].size;
    }

    auto BinaryBroadcast::locate(uint32_t k) const noexcept -> std::pair<uint32_t, uint32_t> {
        uint32_t a = 0, b = 0;
        long rem = k;
        for (auto i : range0_(_strides.size() / 3)) {
            auto d = std::div(rem, _strides[3 * i]);
            a += _strides[3 * i + 1] * d.quot;
            b += _strides[3 * i + 2] * d.quot;
            rem = d.rem;
        }
        return {a, b};
    }
    auto BinaryBroadcast::size() const noexcept -> uint32_t {
        return _size;
    }

    K::BinaryBasicCpu(Op opType_, DT dataType_, BinaryBroadcast b) noexcept
        : Kernel(),
          dataType(dataType_),
          opType(opType_),
          broadcast(std::move(b)) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b) noexcept -> KernelBox {
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
            case Op::Pow: {
                switch (dataType.internal) {
                    CASE_DT(std::pow, F32);
                    CASE_DT(std::pow, F64);
                    CASE_DT(std::pow, I8);
                    CASE_DT(std::pow, I16);
                    CASE_DT(std::pow, I32);
                    CASE_DT(std::pow, I64);
                    default:
                        UNREACHABLE();
                }
            }
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
