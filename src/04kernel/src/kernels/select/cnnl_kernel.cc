#include "cnnl_kernel.hh"

#ifdef USE_BANG
#include "../../utilities/bang/cnnl_context.hh"
#include "../../utilities/bang/cnnl_functions.h"
#endif

namespace refactor::kernel {
    using K = SelectCnnl;

    K::SelectCnnl(decltype(dataType) dataType_,
                  decltype(selectType) selectType_,
                  decltype(inputDims) inputDims_,
                  decltype(outputDims) outputDims_,
                  decltype(inputsNum) inputsNum_) noexcept
        : dataType(dataType_),
          selectType(selectType_),
          inputDims(std::move(inputDims_)),
          outputDims(std::move(outputDims_)),
          inputsNum(inputsNum_) {}

    auto K::build(SelectType selectType_, TensorRefs inputs_) noexcept -> KernelBox {
#ifndef USE_BANG
        return nullptr;
#endif
        auto dt = inputs_[0].get().dataType;
        std::vector<std::vector<int>> inputDims_, outputDims_;
        for (size_t i = 0; i < inputs_.size(); i++) {
            auto shape = std::vector<int>(inputs_[i].get().shape.begin(),
                                          inputs_[i].get().shape.end());
            if (shape.size() == 0) {
                shape.push_back(1);
            }
            inputDims_.push_back(shape);
        }

        auto broadcastShape = [](const std::vector<int> &shape1, const std::vector<int> &shape2) -> std::vector<int> {
            int max_dim = std::max(shape1.size(), shape2.size());

            std::vector<int> resultShape(max_dim, 1);
            int dim_diff1 = max_dim - shape1.size();
            int dim_diff2 = max_dim - shape2.size();

            for (int i = 0; i < max_dim; ++i) {
                int dim_size1 = (i >= dim_diff1) ? shape1[i - dim_diff1] : 1;
                int dim_size2 = (i >= dim_diff2) ? shape2[i - dim_diff2] : 1;
                resultShape[i] = std::max(dim_size1, dim_size2);
            }

            return resultShape;
        };

        for (size_t i = 1; i < inputs_.size(); i++) {
            outputDims_.push_back(broadcastShape(inputDims_[i - 1], inputDims_[i]));
        }

        return std::make_unique<K>(dt, selectType_, inputDims_, outputDims_, inputs_.size());
    }

    auto K::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }

    auto K::kernelTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto K::description() const noexcept -> std::string_view {
        return "Performing select operation using CNNL";
    }

#ifdef USE_BANG
    auto K::lower(Resources &res) const -> RoutineWorkspace {
        using namespace cnnl;
        using namespace runtime;

        struct Descriptors {
            std::vector<cnnlTensorDescriptor_t> in, out;

            explicit Descriptors(int n)
                : in(std::vector<cnnlTensorDescriptor_t>(n, nullptr)),
                  out(std::vector<cnnlTensorDescriptor_t>(n - 1, nullptr)) {
                for (auto i = 0; i < n; i++) {
                    CNNL_ASSERT(cnnlCreateTensorDescriptor(&in[i]));
                    if (i != n - 1) {
                        CNNL_ASSERT(cnnlCreateTensorDescriptor(&out[i]));
                    }
                }
            }
            ~Descriptors() noexcept(false) {
                for (size_t i = 0; i < in.size(); i++) {
                    CNNL_ASSERT(cnnlDestroyTensorDescriptor(in[i]));
                    if (i != in.size() - 1) {
                        CNNL_ASSERT(cnnlDestroyTensorDescriptor(out[i]));
                    }
                }
            }

            Descriptors(const Descriptors &) = delete;
            Descriptors(Descriptors &&) = delete;
        };
        auto d = std::make_shared<Descriptors>(inputsNum);
        for (size_t i = 0; i < inputsNum; i++) {
            setCnnlTensor(d->in[i], dataType, slice(inputDims[i].data(), inputDims[i].size()));
            if (i != inputsNum - 1) {
                setCnnlTensor(d->out[i], dataType, slice(outputDims[i].data(), outputDims[i].size()));
            }
        }

        auto handle = res.fetchOrStore<CnnlContext>()->handle;
        size_t workspaceSize;
        switch (selectType) {
            case SelectType::Max:
                CNNL_ASSERT(cnnlGetMaximumWorkspaceSize(handle, d->out.back(), &workspaceSize));
                break;
            case SelectType::Min:
                CNNL_ASSERT(cnnlGetMinimumWorkspaceSize(handle, d->out.back(), &workspaceSize));
                break;
            default:
                UNREACHABLE();
        }

        res.fetchOrStore<CnnlContext>();
        auto routine = [d = std::move(d), type = selectType, workspaceSize](Resources &res, void *workspace, void const *const *inputs, void *const *outputs) {
            // fetch cnnl handle from resources
            auto handle = res.fetchOrStore<CnnlContext>()->handle;

            auto select =
                (type == SelectType::Max)   ? cnnlMaximum
                : (type == SelectType::Min) ? cnnlMinimum
                                            : nullptr;

            for (size_t i = 1; i < d->in.size(); i++) {
                if (i == 1) {
                    CNNL_ASSERT(select(
                        handle, d->in[0], inputs[0], d->in[1], inputs[1],
                        d->out[0], outputs[0], workspace, workspaceSize));
                } else {
                    CNNL_ASSERT(select(
                        handle, d->out[i - 2], outputs[0], d->in[i], inputs[i],
                        d->out[i - 1], outputs[0], workspace, workspaceSize));
                }
            }
        };

        return {std::move(routine), workspaceSize};
    }

#endif

}// namespace refactor::kernel
