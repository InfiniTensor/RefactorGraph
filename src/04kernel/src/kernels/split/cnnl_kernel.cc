#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include <cnnl.h>
#endif

namespace refactor::kernel {
    using K = SplitCnnl;
    using Info = SplitInfoCnnl;

    Info::SplitInfoCnnl(DataType dt_, int axis_, int num_, std::vector<int> in_, std::vector<std::vector<int>> out_)
        : dataType(dt_), axis(axis_), num(num_), inDim(std::move(in_)), outDims(std::move(out_)) {}


    Info::SplitInfoCnnl(int axis, Tensor const &input, TensorRefs outputs)
        : SplitInfoCnnl(input.dataType, axis, outputs.size(),
                        std::move(std::vector<int>(input.shape.begin(), input.shape.end())),
                        std::move([](TensorRefs tensors) -> std::vector<std::vector<int>> {
                            std::vector<std::vector<int>> res;
                            for (uint32_t i = 0; i < tensors.size(); i++) {
                                res.push_back(std::vector<int>(tensors[i].get().shape.begin(),
                                                               tensors[i].get().shape.end()));
                            }
                            return res;
                        }(outputs))) {}

    K::SplitCnnl(SplitInfoCnnl info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(int axis, Tensor const &input, TensorRefs outputs) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif
        return std::make_unique<K>(SplitInfoCnnl(axis, input, outputs));
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing concat operation using CNNL";
    }

#ifdef USE_BANG
    auto SplitCnnl::lower(Resources &res) const noexcept -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;
        using DT = DataType;

        struct Descriptors {
            cnnlTensorDescriptor_t in;
            std::vector<cnnlTensorDescriptor_t> out;
            bool f32;

            explicit Descriptors(int n, decltype(f32) f32_)
                : in(nullptr),
                  out(std::vector<cnnlTensorDescriptor_t>(n, nullptr)),
                  f32(f32_) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&in));
                for (auto i = 0; i < n; i++) {
                    CNNL_ASSERT(cnnlCreateTensorDescriptor(&out[i]));
                }
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(in));
                for (size_t i = 0; i < out.size(); i++) {
                    CNNL_ASSERT(cnnlDestroyTensorDescriptor(out[i]));
                }
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>(info.num, info.dataType != DT::F64);
        // setCnnlTensor(d->in, info.dataType, slice(info.inDim.data(), info.inDim.size()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->in, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(info.dataType), info.inDim.size(), info.inDim.data()));

        for (size_t i = 0; i < info.outDims.size(); i++) {
            // setCnnlTensor(d->out[i], info.dataType, slice(info.outDims[i].data(), info.outDims[i].size()));
            CNNL_ASSERT(cnnlSetTensorDescriptor(d->out[i], CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(info.dataType), info.outDims[i].size(), info.outDims[i].data()));
        }

        auto handle = res.fetchOrStore<CnnlContext>()->handle;
        size_t workspaceSize;
        CNNL_ASSERT(cnnlGetSplitWorkspaceSize(handle, info.num, &workspaceSize));

        res.fetchOrStore<CnnlContext>();
        auto routine = [d = std::move(d), n = info.num, axis = info.axis, workspaceSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            // fetch cnnl handle from resources
            auto handle = res.fetchOrStore<CnnlContext>()->handle;

            void *argv[n];
            for (auto i = 0; i < n; i++) {
                argv[i] = outputs[i];
            }

            CNNL_ASSERT(cnnlSplit(
                handle, n, axis, d->in, inputs[0],
                workspace, workspaceSize, d->out.data(), argv));
        };

        return {std::move(routine), workspaceSize};
    }

#endif


}// namespace refactor::kernel
