#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#endif

namespace refactor::kernel {
    using K = PadCnnl;

    K::PadCnnl(DataType dataType_, PadType mode_, std::vector<int> inDim_,
               std::vector<int> outDim_, std::vector<int> padDim_, size_t len_) noexcept
        : Kernel(), dataType(dataType_), mode(mode_), inDim(std::move(inDim_)),
          outDim(std::move(outDim_)), padDim(std::move(padDim_)), valueLength(len_) {}

    auto K::build(PadDimension dims_, DataType dataType_, PadType mode_, std::optional<std::reference_wrapper<Tensor const>> value_) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif
        if (mode_ != PadType::Constant || (value_ && value_->get().dataType != dataType_)) {
            return nullptr;
        }
        size_t valueLength_ = value_ ? value_->get().dataType.size() : 0;
        std::vector<int> inDim_, outDim_, padDim_;
        for (auto dim : dims_) {
            inDim_.push_back(dim.dimI);
            outDim_.push_back(dim.dimO);
            padDim_.push_back(dim.pads);
        }

        return std::make_unique<K>(dataType_, mode_, inDim_, outDim_, padDim_, valueLength_);
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t { return typeId(); }
    auto K::description() const noexcept -> std::string_view {
        return "Performing Pad using CNNL";
    }

#ifdef USE_BANG
    auto K::lower(Resources &res) const noexcept -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;

        struct Descriptors {
            cnnlTensorDescriptor_t inDesc, outDesc;

            Descriptors() : inDesc(nullptr), outDesc(nullptr) {
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&inDesc));
                CNNL_ASSERT(cnnlCreateTensorDescriptor(&outDesc));
            }
            ~Descriptors() noexcept(false) {
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(inDesc));
                CNNL_ASSERT(cnnlDestroyTensorDescriptor(outDesc));
            }
        };
        auto d = std::make_shared<Descriptors>();
        setCnnlTensor(d->inDesc, dataType, slice(inDim.data(), inDim.size()));
        setCnnlTensor(d->outDesc, dataType, slice(outDim.data(), outDim.size()));

        std::vector<int> pads;
        for (auto d : padDim) {
            pads.push_back(d);
            pads.push_back(d);
        }

        res.fetchOrStore<CnnlContext>();
        return [d = std::move(d), val = valueLength,
                p = std::vector<int>(pads.begin(), pads.end())](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            void *paddingValue;
            if (val != 0) {
                paddingValue = malloc(val);
                BANG_ASSERT(cnrtMemcpy(paddingValue, const_cast<void *>(inputs[2]),
                                       val, CNRT_MEM_TRANS_DIR_DEV2HOST));
            } else {
                float zero = 0.0;
                paddingValue = &zero;
            }

            CNNL_ASSERT(cnnlPad(res.fetchOrStore<CnnlContext>()->handle,
                                d->inDesc, inputs[0], p.data(), paddingValue,
                                d->outDesc, outputs[0]));

            if (val != 0) {
                free(paddingValue);
            }
        };
    }
#endif

}// namespace refactor::kernel
