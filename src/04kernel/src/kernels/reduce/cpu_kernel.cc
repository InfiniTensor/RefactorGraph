#include "cpu_kernel.hh"
#include "runtime/mem_manager.hh"
#include <numeric>

namespace refactor::kernel {
    using K = ReduceCpu;
    using DT = DataType;

    K::ReduceCpu(decltype(axes) axes_, ReduceType reduceType_, DataType dataType_, Shape shape_) noexcept
        : Kernel(), axes(axes_), reduceType(reduceType_), dataType(dataType_), shape(shape_) {}

    auto K::build(decltype(axes) axes_, ReduceType reduceType_, TensorRefs inputs_) noexcept -> KernelBox {
        auto const &x = inputs_[0].get();
        return std::make_unique<K>(axes_, reduceType_, x.dataType, x.shape);
    }
    auto K::typeId() noexcept -> size_t {
        // 这里 ID 是什么含义？
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing reduce operation on generic cpu";
    }

    template<decltype(DT::internal) T>
    Routine lowerReduceMean(Shape shape, std::vector<int64_t> axes) {
        using namespace runtime;
        using dt = typename primitive_t<T>::type;
        return [shape, axes](Resources &res, void const **inputs, void **outputs) {
            std::vector<int64_t> axes_(axes.begin(), axes.end());
            std::sort(axes_.begin(), axes_.end());
            // 计算最大的 axis 轴之前的总维数
            size_t outside = 1;
            for (size_t i = 0; i < *(axes_.end()); ++i) {
                outside *= shape[i];
            }
            // 计算最大的 axis 轴之后的总维数
            size_t inside = 1;
            for (size_t i = *(axes_.end()) + 1; i < shape.size(); ++i) {
                inside *= shape[i];
            }
            constexpr static auto workspaceSize = 4ul << 30;
            auto workspace = mem_manager::ForeignBlob::share(res.fetch<runtime::MemManager>()->manager, workspaceSize);
            auto src = reinterpret_cast<dt const *>(inputs[0]);
            auto dst = reinterpret_cast<dt *>(outputs[0]);
            // 按照从大到小的顺序依次 reduce axes_ 里的每个轴
            for (auto axesIter = axes_.rbegin(); axesIter != axes_.rend(); ++axesIter) {
                if (axesIter != axes_.rend() - 1) {
                    dst = reinterpret_cast<dt *>((void *) *workspace);
                } else {
                    dst = reinterpret_cast<dt *>(outputs[0]);
                }
                // 第一个 for 循环，遍历所有输入的高维矩阵
                for (size_t oi = 0; oi < outside; ++oi) {
                    auto axisVal = *axesIter;
                    // 计算 src 中第 oi 个高维矩阵的起始地址
                    auto srcOutside = src + oi * axisVal * inside;
                    // 计算 dst 中存放第 oi 个高维矩阵 mean 计算结果的起始地址
                    auto dstOutside = dst + oi * inside;
                    // 第二个 for 循环，遍历第 oi 个高维矩阵在 axis 轴之后所有的维数
                    for (size_t ii = 0; ii < inside; ++ii) {
                        auto srcInside = srcOutside + ii;
                        auto dstInside = dstOutside + ii;
                        dt summer = 0;
                        // 第三个 for 循环，遍历 axis 轴上的每个元素做 sum 计算
                        for (size_t a = 0; a < axisVal; ++a) {
                            summer += srcInside[a * inside];
                        }
                        // 汇总计算 mean 值
                        *dstInside = summer / axisVal;
                    }
                }
                src = dst;
                outside /= *axesIter;
            }
        };
    }

    auto K::lower() const noexcept -> Routine {
        // TODO: 根据 reduceType 选择 lower 函数

#define CASE(T) \
    case T:     \
        return lowerReduceMean<DataType::T>(shape, axes)

        switch (dataType) {
            CASE(DataType::U32);
            CASE(DataType::U64);
            CASE(DataType::I32);
            CASE(DataType::I64);
            // CASE(DataType::FP16);
            CASE(DataType::F32);
            CASE(DataType::F64);
            // CASE(DataType::BF16);
            default:
                UNREACHABLE();
        }
    }

}// namespace refactor::kernel
