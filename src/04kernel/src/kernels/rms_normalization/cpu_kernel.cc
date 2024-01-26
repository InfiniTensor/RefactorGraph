#include "cpu_kernel.hh"
#include <execution>
#include <numeric>

namespace refactor::kernel {
    using K = RmsNormalizationCpu;

    K::RmsNormalizationCpu(
        decltype(epsilon) epsilon_,
        decltype(dataType) dataType_,
        decltype(blockCount) blockCount_,
        decltype(blockSize) blockSize_) noexcept
        : Kernel(),
          epsilon(epsilon_),
          dataType(dataType_),
          blockCount(blockCount_),
          blockSize(blockSize_) {}

    auto K::build(float epsilon, TensorRefs inputs) noexcept -> KernelBox {
        auto const &x = inputs[0].get();
        auto const &w = inputs[1].get();
        if ((x.dataType != DataType::F32 && x.dataType != DataType::F64) || x.dataType != w.dataType) {
            return nullptr;
        }
        auto it = x.shape.rbegin();
        dim_t blockSize = *it++;
        dim_t blockCount = std::accumulate(it, x.shape.rend(), 1, std::multiplies());
        return std::make_unique<K>(epsilon, x.dataType, blockCount, blockSize);
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing rms normalization on generic cpu";
    }

    template<class T>
    static Routine lowerTyped(float epsilon, dim_t blockCount, dim_t blockSize) {
        using namespace runtime;

        return [epsilon, blockCount, blockSize]//
            (Resources &, void *, void const *const *inputs, void *const *outputs) {
                auto x = reinterpret_cast<T const *>(inputs[0]);
                auto w = reinterpret_cast<T const *>(inputs[1]);
                auto y = reinterpret_cast<T *>(outputs[0]);
                std::for_each_n(
                    std::execution::par_unseq,
                    natural_t(0),
                    blockCount,
                    [blockSize, epsilon, x, w, y](auto i) {
                        auto x_ = x + i * blockSize;
                        auto y_ = y + i * blockSize;

                        auto ss = std::accumulate(
                            x_, x_ + blockSize, 0,
                            [](auto acc, auto it) { return acc + it * it; });
                        ss /= blockSize;
                        ss += epsilon;
                        ss = 1. / std::sqrt(ss);

                        for (auto j : range0_(blockSize)) {
                            y_[j] = x_[j] * ss * w[j];
                        }
                    });
            };
    }

    auto K::lower(Resources &) const noexcept -> RoutineWorkspace {
        return dataType == DataType::F32
                   ? lowerTyped<float>(epsilon, blockCount, blockSize)
                   : lowerTyped<double>(epsilon, blockCount, blockSize);
    }

}// namespace refactor::kernel
