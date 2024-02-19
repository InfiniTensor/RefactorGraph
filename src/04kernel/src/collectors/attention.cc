#include "kernel/collectors/attention.h"
// #include "../kernels/attention/cpu_kernel.hh"
#include "../kernels/attention/cuda_kernel.hh"

namespace refactor::kernel {

    AttentionCollector::AttentionCollector(
        decltype(_target) target,
        decltype(maxSeqLen) maxSeqLen_) noexcept
        : InfoCollector(target),
          maxSeqLen(maxSeqLen_) {}

    std::vector<KernelBox>
    AttentionCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &query = inputs[0].get();
        auto const &key = inputs[1].get();
        auto pastSeqLen = inputs.size() == 3 ? 0 : *inputs[2].get().data->get<int64_t>();
        auto cacheLen = outputs.size() == 1 ? 0 : outputs[1].get().shape[2];

        std::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                break;
            case decltype(_target)::Nvidia: {
                decltype(AttentionCuda::info) info{
                    .dataType = query.dataType,
                    .batch = query.shape[0],
                    .nHead = query.shape[1],
                    .nKVHead = key.shape[1],
                    .pastSeqLen = static_cast<dim_t>(pastSeqLen),
                    .seqLen = query.shape[2],
                    .cacheLen = cacheLen,
                    .headDim = query.shape[3],
                    .resetCache = false,
                };
                if (auto ptr = AttentionCuda::build(info); ptr) {
                    ans.emplace_back(std::move(ptr));
                }
            } break;
            case decltype(_target)::Mlu:
                break;
            default:
                UNREACHABLEX(void, "Unknown target");
        }
        return ans;
    }

}// namespace refactor::kernel
