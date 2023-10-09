#include "arthimetic11_cuda.hh"
#include "common/error_handler.h"
#include "common/natural.h"
#include <execution>
#include <unordered_set>

#ifdef USE_CUDA
#include "arthimetic_cuda.h"
#endif

namespace refactor::kernel {
    using K = Arthimetic11Cuda;
    using Op = SimpleBinaryType;
    using DT = common::DataType;

    K::Arthimetic11Cuda(Op opType_, DT dataType_, size_t size_) noexcept
        : Kernel(), dataType(dataType_), opType(opType_), size(size_) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b, Tensor const &c) noexcept -> KernelBox {
        static const std::unordered_set<decltype(DT::internal)> TYPE{
            DT::F32, DT::U8, DT::I8, DT::U16, DT::I16,
            DT::I32, DT::I64, DT::F64, DT::U32, DT::U64};
        static const std::unordered_set<Op> ARTHIMETIC{
            Op::Add, Op::Sub, Op::Mul, Op::Div};

#ifndef USE_CUDA
        return nullptr;
#endif

        auto dataType = a.dataType;
        if (a.shape != b.shape ||
            a.shape != c.shape ||
            dataType != b.dataType ||
            dataType != c.dataType ||
            ARTHIMETIC.find(op) == ARTHIMETIC.end() ||
            TYPE.find(dataType.internal) == TYPE.end()) {
            return nullptr;
        }
        return std::make_unique<K>(op, dataType, a.elementsSize());
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing add/sub/mul/div of 2 tensors of same shape on nvidia gpu";
    }

    auto K::lower() const noexcept -> Operation {
        TODO("");
    }

}// namespace refactor::kernel
