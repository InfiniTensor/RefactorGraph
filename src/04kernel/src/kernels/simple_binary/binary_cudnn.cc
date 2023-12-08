#include "binary_cudnn.hh"
#include <unordered_set>

#ifdef USE_CUDA
#include "../../utilities/cuda/cudnn_context.hh"
#include "../../utilities/cuda/cudnn_functions.h"
#endif

namespace refactor::kernel {
    using K = BinaryCudnn;
    using Op = SimpleBinaryType;
    using DT = DataType;

    K::BinaryCudnn(Op opType_, DT dataType_, std::vector<int> aDims_, std::vector<int> bDims_, std::vector<int> cDims_) noexcept
        : Kernel(), dataType(dataType_), opType(opType_), aDims(aDims_), bDims(bDims_), cDims(cDims_) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b, Tensor const &c) noexcept -> KernelBox {
        static const std::unordered_set<Op>
            ARTHIMETIC{Op::Add, Op::Sub, Op::Mul};

#ifndef USE_CUDA
        return nullptr;
#endif

        if (a.dataType != b.dataType ||
            !a.dataType.isFloat() ||
            !ARTHIMETIC.contains(op) ||
            // At least one of a,b should have the same shape as c
            (a.shape != c.shape && b.shape != c.shape) ||
            // Sub only supports brocasting b
            (a.shape != c.shape && op == Op::Sub) ||
            // Cudnn binary op only supports up to 5D
            !((a.rank() == 5 && b.rank() == 5) || (a.rank() <= 4 && b.rank() <= 4))) {
            return nullptr;
        }

        auto shape2IntVec = [](Shape shape) -> std::vector<int> {
            std::vector<int> intVector;
            intVector.reserve(shape.size());
            for (const uint32_t &element : shape) {
                intVector.push_back(static_cast<int>(element));
            }
            return intVector;
        };

        return std::make_unique<K>(op, a.dataType, shape2IntVec(a.shape), shape2IntVec(b.shape), shape2IntVec(c.shape));
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing element-wise op of 2 tensors with CUDNN";
    }

#ifdef USE_CUDA

    auto BinaryCudnn::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cudnn;
        using namespace runtime;

        struct Descriptors {
            cudnnOpTensorDescriptor_t opDesc;
            cudnnTensorDescriptor_t aDesc, bDesc, cDesc;
            float aAlpha = 1.f;
            float bAlpha = 1.f;
            float beta = 0.f;
            Descriptors() {
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&aDesc));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&bDesc));
                CUDNN_ASSERT(cudnnCreateTensorDescriptor(&cDesc));
                CUDNN_ASSERT(cudnnCreateOpTensorDescriptor(&opDesc));
            }
            ~Descriptors() noexcept(false) {
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(aDesc));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(bDesc));
                CUDNN_ASSERT(cudnnDestroyTensorDescriptor(cDesc));
                CUDNN_ASSERT(cudnnDestroyOpTensorDescriptor(opDesc));
            }
        };
        auto d = std::make_shared<Descriptors>();
        cudnnOpTensorOp_t cudnnOP;
        if (opType == SimpleBinaryType::Add) {
            cudnnOP = CUDNN_OP_TENSOR_ADD;
        } else if (opType == SimpleBinaryType::Sub) {
            cudnnOP = CUDNN_OP_TENSOR_ADD;
            d->bAlpha = -1.f;
        } else if (opType == SimpleBinaryType::Mul) {
            cudnnOP = CUDNN_OP_TENSOR_MUL;
        } else {
            UNREACHABLE();
        }

        setCudnnTensor(d->aDesc, dataType, slice(aDims.data(), aDims.size()));
        setCudnnTensor(d->bDesc, dataType, slice(bDims.data(), bDims.size()));
        setCudnnTensor(d->cDesc, dataType, slice(cDims.data(), cDims.size()));
        CUDNN_ASSERT(cudnnSetOpTensorDescriptor(
            d->opDesc, cudnnOP, cudnnDataTypeConvert(dataType), CUDNN_NOT_PROPAGATE_NAN));

        res.fetchOrStore<CudnnContext>();
        return [swap = aDims != cDims,
                d_ = std::move(d)](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            auto handle = res.fetchOrStore<CudnnContext>()->handle;
            auto const &d = *d_;
            // name inputs and outputs
            auto a = inputs[0],
                 b = inputs[1];
            auto c = outputs[0];
            if (swap) {
                CUDNN_ASSERT(cudnnOpTensor(handle, d.opDesc, &(d.aAlpha),
                                           d.bDesc, b, &(d.bAlpha), d.aDesc, a,
                                           &(d.beta), d.cDesc, c));
            } else {
                CUDNN_ASSERT(cudnnOpTensor(handle, d.opDesc, &(d.aAlpha),
                                           d.aDesc, a, &(d.bAlpha), d.bDesc, b,
                                           &(d.beta), d.cDesc, c));
            }
        };
    }

#endif

}// namespace refactor::kernel
