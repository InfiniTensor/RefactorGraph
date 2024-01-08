#include "binary_cnnl.hh"
#include <unordered_set>

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#endif

namespace refactor::kernel {
    using K = BinaryCnnl;
    using Op = SimpleBinaryType;
    using DT = DataType;

    K::BinaryCnnl(Op opType_, DT dataType_, std::vector<int> aDims_, std::vector<int> bDims_, std::vector<int> cDims_) noexcept
        : Kernel(), dataType(dataType_), opType(opType_), aDims(aDims_), bDims(bDims_), cDims(cDims_) {}

    auto K::build(Op op, Tensor const &a, Tensor const &b, Tensor const &c) noexcept -> KernelBox {
        static const std::unordered_set<Op>
            ARTHIMETIC{Op::Add, Op::Sub, Op::Mul};

#ifndef USE_BANG
        return nullptr;
#endif

        if (a.dataType != b.dataType ||
            !a.dataType.isFloat() ||
            !ARTHIMETIC.contains(op) ||
            // At least one of a,b should have the same shape as c
            (a.shape != c.shape && b.shape != c.shape) ||
            // Sub only supports brocasting b
            (a.shape != c.shape && op == Op::Sub) ||
            // Cnnl binary op only supports up to 5D
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
        return "Performing element-wise op of 2 tensors with CNNL";
    }

#ifdef USE_BANG

    auto BinaryCnnl::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;

        struct Descriptors {
            cnnlOpTensorDescriptor_t opDesc;
            cnnlTensorDescriptor_t aDesc, bDesc, cDesc;
            bool f32, sub;

            Descriptors(decltype(f32) f32_) : f32(f32_), sub(false) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&aDesc));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&bDesc));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&cDesc));
                CNNL_ASSERT(cnnlCreateOpTensorDescriptor(&opDesc));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(aDesc));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(bDesc));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(cDesc));
                CNNL_ASSERT(cnnlDestroyOpTensorDescriptor(opDesc));
            }
        };
        auto d = std::make_shared<Descriptors>(dataType != DT::F64);
        cnnlOpTensorDesc_t cnnlOP;
        if (opType == SimpleBinaryType::Add) {
            cnnlOP = CNNL_OP_TENSOR_ADD;
        } else if (opType == SimpleBinaryType::Sub) {
            cnnlOP = CNNL_OP_TENSOR_ADD;
            d->sub = true;
        } else if (opType == SimpleBinaryType::Mul) {
            cnnlOP = CNNL_OP_TENSOR_MUL;
        } else {
            UNREACHABLE();
        }

        setCnnlTensor(d->aDesc, dataType, slice(aDims.data(), aDims.size()));
        setCnnlTensor(d->bDesc, dataType, slice(bDims.data(), bDims.size()));
        setCnnlTensor(d->cDesc, dataType, slice(cDims.data(), cDims.size()));
        CNNL_ASSERT(cnnlSetOpTensorDescriptor(
            d->opDesc, cnnlOP,
            cnnlDataTypeConvert(d->f32 ? DT::F32 : DT::F64),
            CNNL_NOT_PROPAGATE_NAN));

        return [swap = aDims != cDims, d](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            auto handle = res.fetchOrStore<CnnlContext>()->handle;
            // name inputs and outputs
            auto a = inputs[0],
                 b = inputs[1];
            auto c = outputs[0];
            auto alphaA = d->f32
                              ? factor<fp32_t>(1)
                              : factor<fp64_t>(1),
                 alphaB = d->f32
                              ? factor<fp32_t>(d->sub ? -1 : 1)
                              : factor<fp64_t>(d->sub ? -1 : 1),
                 beta = d->f32
                            ? factor<fp32_t>(0)
                            : factor<fp64_t>(0);
            size_t workspaceSize;
            if (swap) {
                CNNL_ASSERT(cnnlGetOpTensorWorkspaceSize(handle, d->bDesc,
                                                         d->aDesc, d->cDesc,
                                                         &workspaceSize));
                CNNL_ASSERT(cnnlOpTensor(handle, d->opDesc,
                                         &alphaB, d->bDesc, b,
                                         &alphaA, d->aDesc, a,
                                         workspace, workspaceSize,
                                         &beta, d->cDesc, c));
            } else {
                CNNL_ASSERT(cnnlGetOpTensorWorkspaceSize(handle, d->aDesc,
                                                         d->bDesc, d->cDesc,
                                                         &workspaceSize));
                CNNL_ASSERT(cnnlOpTensor(handle, d->opDesc,
                                         &alphaA, d->aDesc, a,
                                         &alphaB, d->bDesc, b,
                                         workspace, workspaceSize,
                                         &beta, d->cDesc, c));
            }
        };
    }

#endif

}// namespace refactor::kernel
