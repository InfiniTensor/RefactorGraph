#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#include <cnnl.h>
#endif

namespace refactor::kernel {
    using K = SliceCnnl;

    K::SliceCnnl(decltype(info) info_) noexcept
        : Kernel(), info(std::move(info_)) {}

    auto K::build(DataType dt_, Dimensions dims_, Shape in_, Shape out_) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif
        return std::make_unique<K>(decltype(info){
            dt_,
            dims_,
            std::vector<int>(in_.begin(), in_.end()),
            std::vector<int>(out_.begin(), out_.end()),
        });
    }
    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing slice operation using CNNL";
    }

#ifdef USE_BANG
    auto SliceCnnl::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;
        using DT = DataType;

        struct Descriptors {
            cnnlTensorDescriptor_t in, out;
            bool f32;

            explicit Descriptors(decltype(f32) f32_)
                : in(nullptr), out(nullptr), f32(f32_) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&in));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&out));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(in));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(out));
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>(info.dataType != DT::F64);
        // setCnnlTensor(d->in, info.dataType, slice(info.inDim.data(), info.inDim.size()));
        // setCnnlTensor(d->out, info.dataType, slice(info.outDim.data(), info.outDim.size()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->in, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(info.dataType), info.inDim.size(), info.inDim.data()));
        CNNL_ASSERT(cnnlSetTensorDescriptor(d->out, CNNL_LAYOUT_NCHW, cnnlDataTypeConvert(info.dataType), info.outDim.size(), info.outDim.data()));
        std::vector<int> begin, end, stride;
        for (size_t i = 0; i < info.dims.size(); i++) {
            // [begin, end), end is not inclued
            begin.push_back(info.dims[i].start);
            auto sign = info.dims[i].step > 0 ? 1 : -1;
            end.push_back(info.dims[i].start + info.dims[i].step * (info.dims[i].length - 1) + sign);
            stride.push_back(info.dims[i].step);
        }

        res.fetchOrStore<CnnlContext>();
        return [d = std::move(d), begin, end, stride](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            // fetch cnnl handle from resources
            auto handle = res.fetchOrStore<CnnlContext>()->handle;

            CNNL_ASSERT(cnnlStridedSlice(
                handle, d->in, inputs[0],
                begin.data(), end.data(), stride.data(),
                d->out, outputs[0]));
        };
    }
#endif

}// namespace refactor::kernel
