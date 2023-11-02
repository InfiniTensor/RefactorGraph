#include "cpu_kernel.hh"
#include <numeric>

namespace refactor::kernel {
    using K = BatchNormalization;
    using DT = DataType;

    K::BatchNormalization(
        float epsilon_,
        DT dt0,
        DT dt1,
        DT dt2,
        Shape shape_) noexcept
        : Kernel(),
          epsilon(epsilon_),
          dts{dt0, dt1, dt2},
          shape(std::move(shape_)) {}

    auto K::build(float epsilon, TensorRefs inputs) noexcept -> KernelBox {
        auto const &x = inputs[0].get();
        auto const &scale = inputs[1].get();
        auto const &mean = inputs[3].get();
        if (!x.dataType.isCpuNumberic() ||
            !scale.dataType.isCpuNumberic() ||
            !mean.dataType.isCpuNumberic()) {
            return nullptr;
        }
        return std::make_unique<K>(epsilon, x.dataType, scale.dataType, mean.dataType, x.shape);
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing batch normalization for non-training-mode on generic cpu";
    }

    template<decltype(DT::internal) T, decltype(DT::internal) T1, decltype(DT::internal) T2>
    Routine lowerTyped(Shape const &shape, float epsilon) {
        using namespace runtime;
        using dt = typename primitive<T>::type;
        using t1 = typename primitive<T1>::type;
        using t2 = typename primitive<T2>::type;

        auto n = shape[0],
             c = shape[1],
             dims = std::accumulate(shape.begin() + 2, shape.end(), 1u, std::multiplies<>()),
             sn = c * dims,
             sc = dims;
        return [n, c, sn, sc, epsilon](Resources &, void const **inputs, void **outputs) {
            auto x = inputs[0],
                 scale = inputs[1],
                 bias = inputs[2],
                 mean = inputs[3],
                 var = inputs[4];
            auto y = outputs[0];

            struct Channel {
                dt mean, scale, bias;
            };
            std::vector<Channel> channels(c);
            auto scale_ = reinterpret_cast<t1 const *>(scale),
                 bias_ = reinterpret_cast<t1 const *>(bias);
            auto mean_ = reinterpret_cast<t2 const *>(mean),
                 var_ = reinterpret_cast<t2 const *>(var);
            for (auto i : range0_(c)) {
                channels[i] = {
                    static_cast<dt>(mean_[i]),
                    static_cast<dt>(scale_[i]) / std::sqrt(static_cast<dt>(var_[i]) + epsilon),
                    static_cast<dt>(bias_[i]),
                };
            }
            // Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
            auto x_ = reinterpret_cast<dt const *>(x),
                 y_ = reinterpret_cast<dt *>(y);
            for (auto in : range0_(n))
                for (auto ic : range0_(c))
                    for (auto j : range0_(sc)) {
                        auto idx = in * sn + ic * sc + j;
                        auto [_, a, b] = channels[ic];
                        y_[idx] = (x_[idx] - _) * a + b;
                    }
        };
    }

    auto K::lower() const noexcept -> Routine {
        // clang-format off
        static_assert(sizeof(decltype(DT::internal)) == 1);
        #define MERGE(DT0, DT1, DT2)                      \
                  (static_cast<uint32_t>(DT0)           ) \
                + (static_cast<uint32_t>(DT1) << (1 * 8)) \
                + (static_cast<uint32_t>(DT2) << (2 * 8))

        #define CASE(DT0, DT1, DT2)                \
            case MERGE(DT::DT0, DT::DT1, DT::DT2): \
                return lowerTyped<DT::DT0, DT::DT1, DT::DT2>(shape, epsilon)

        switch (MERGE(dts[0], dts[1], dts[2])) {
            CASE(F32, F32, F32);
            CASE(F32, F32, F64);
            CASE(F32, F64, F32);
            CASE(F32, F64, F64);
            CASE(F64, F32, F32);
            CASE(F64, F32, F64);
            CASE(F64, F64, F32);
            CASE(F64, F64, F64);
            default: UNREACHABLE();
        }
        // clang-format on
    }

}// namespace refactor::kernel
