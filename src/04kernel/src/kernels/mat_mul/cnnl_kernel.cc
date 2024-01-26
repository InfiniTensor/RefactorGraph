#include "cnnl_kernel.hh"
#include <numeric>

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include <cnnl.h>
#endif

namespace refactor::kernel {
    using K = MatMulCnnl;
    using DT = DataType;

    K::MatMulCnnl(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(TensorRefs inputs_, TensorRefs outputs_, bool transA_, bool transB_, float alpha_, float beta_) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif
        auto dt = inputs_[0].get().dataType;
        return dt.isIeee754() || dt == DT::I8
                   ? std::make_unique<K>(decltype(info){
                         dt,
                         transA_,
                         transB_,
                         alpha_,
                         beta_,
                         std::vector<int>(inputs_[0].get().shape.begin(), inputs_[0].get().shape.end()),
                         std::vector<int>(inputs_[1].get().shape.begin(), inputs_[1].get().shape.end()),
                         std::vector<int>(outputs_[0].get().shape.begin(), outputs_[0].get().shape.end()),
                         inputs_.size() == 3
                             ? inputs_[2].get().shape.size() == 0 ? std::make_optional(std::vector<int>(1, 1))
                                                                  : std::make_optional(std::vector<int>(
                                                                        inputs_[2].get().shape.begin(),
                                                                        inputs_[2].get().shape.end()))
                             : std::nullopt,
                     })
                   : nullptr;
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing MatMul using CNNL";
    }


#ifdef USE_BANG
    auto K::lower(Resources &res) const noexcept -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;
        using DT = DataType;

        // RAII for closure
        struct Descriptors {
            cnnlTensorDescriptor_t a, b, c;
            cnnlMatMulDescriptor_t bmm;
            cnnlMatMulAlgo_t algo;
            cnnlMatMulHeuristicResult_t heuristic;
            cnnlTensorDescriptor_t bias;
            bool addBias, f32;

            explicit Descriptors(bool addBias_, bool f32_)
                : a(nullptr), b(nullptr), c(nullptr),
                  bmm(nullptr), algo(nullptr), heuristic(nullptr),
                  bias(nullptr), addBias(addBias_), f32(f32_) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&a));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&b));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&c));
                if (addBias) {
                    CNNL_ASSERT(cnnlCreateTensorDescriptor(&bias));
                }
                CNNL_ASSERT(cnnlMatMulDescCreate(&bmm));
                CNNL_ASSERT(cnnlMatMulAlgoCreate(&algo));
                CNNL_ASSERT(cnnlCreateMatMulHeuristicResult(&heuristic));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(a));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(b));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(c));
                if (addBias) {
                    CNNL_ASSERT(cnnlDestroyTensorDescriptor(bias));
                }
                CNNL_ASSERT(cnnlMatMulDescDestroy(bmm));
                CNNL_ASSERT(cnnlMatMulAlgoDestroy(algo));
                CNNL_ASSERT(cnnlDestroyMatMulHeuristicResult(heuristic));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>(info.biasDim.has_value(), info.dataType != DT::F64);
        setCnnlTensor(d->a, info.dataType, slice(info.aDim.data(), info.aDim.size()));
        setCnnlTensor(d->b, info.dataType, slice(info.bDim.data(), info.bDim.size()));
        setCnnlTensor(d->c, info.dataType, slice(info.cDim.data(), info.cDim.size()));
        if (d->addBias) {
            CNNL_ASSERT(cnnlSetTensorDescriptor(
                d->bias, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(info.dataType),
                info.biasDim.value().size(), info.biasDim.value().data()));
        }
        int32_t tA = info.transA, tB = info.transB;
        CNNL_ASSERT(cnnlSetMatMulDescAttr(d->bmm, CNNL_MATMUL_DESC_TRANSA,
                                          &tA, sizeof(int32_t)));
        CNNL_ASSERT(cnnlSetMatMulDescAttr(d->bmm, CNNL_MATMUL_DESC_TRANSB,
                                          &tB, sizeof(int32_t)));
        auto handle = res.fetchOrStore<CnnlContext>()->handle;
        int returnedAlgoCount = 0;
        CNNL_ASSERT(cnnlGetBatchMatMulAlgoHeuristic(
            handle, d->bmm, d->a, d->b, d->c,
            NULL, 1, &(d->heuristic), &returnedAlgoCount));

        size_t algoWorkspaceSize;
        CNNL_ASSERT(cnnlGetBatchMatMulHeuristicResult(d->heuristic, d->algo, &algoWorkspaceSize));

        res.fetchOrStore<CnnlContext>();
        auto routine = [d = std::move(d), algoWorkspaceSize,
                        aa = info.alpha, bb = info.beta](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            // fetch cnnl handle from resources
            auto handle = res.fetchOrStore<CnnlContext>()->handle;

            // build alpha/beta for double
            auto alpha = d->f32 ? factor<fp32_t>(aa) : factor<fp64_t>(aa),
                 beta = d->f32 ? factor<fp32_t>(bb) : factor<fp64_t>(bb),
                //  one = d->f32 ? factor<fp32_t>(1) : factor<fp64_t>(1),
                 zero = d->f32 ? factor<fp32_t>(0) : factor<fp64_t>(0);

            if (d->addBias) {
                CNNL_ASSERT(cnnlExpand(handle, d->bias, inputs[2], d->c, outputs[0]));
            }

            if (alpha != 0) {
                CNNL_ASSERT(cnnlBatchMatMulBCast_v2(
                    handle, d->bmm, d->algo, &alpha,
                    d->a, inputs[0], d->b, inputs[1],
                    d->addBias ? &beta : &zero, d->c, outputs[0],
                    workspace, algoWorkspaceSize));
            }

        };

        return {std::move(routine), algoWorkspaceSize};
    }


#endif

}// namespace refactor::kernel
