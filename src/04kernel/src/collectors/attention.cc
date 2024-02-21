#include "kernel/collectors/attention.h"
#include "kernel/attributes/attention_info.h"
// #include "../kernels/attention/cpu_kernel.hh"
#include "../kernels/attention/cuda_kernel.hh"

namespace refactor::kernel {

    AttentionCollector::AttentionCollector(
        decltype(_target) target) noexcept
        : InfoCollector(target) {}

    std::vector<KernelBox>
    AttentionCollector::filter(TensorRefs inputs, TensorRefs outputs) const {
        auto const &query = inputs[0].get();
        auto const &key = inputs[1].get();

        AttentionInfo info{
            .dataType = query.dataType,
            .batch = query.shape[0],
            .nHead = query.shape[1],
            .nKVHead = key.shape[1],
            .seqLen = query.shape[2],
            .headDim = query.shape[3],
            .cacheLen = 0,
            .concatCache = false,
            .resetCache = false,
        };
        switch (outputs.size()) {
            case 1:
                // no kv cache
                ASSERT(inputs.size() == 3, "");
                break;
            case 3:
                switch (inputs.size()) {
                    case 6:
                        info.resetCache = true;
                    case 4:
                        info.concatCache = true;
                    case 3:
                        info.cacheLen = outputs[1].get().shape[2];
                        break;
                    default:
                        UNREACHABLE();
                }
                break;
            default:
                UNREACHABLE();
        }

        std ::vector<KernelBox> ans;
        switch (_target) {
            case decltype(_target)::Cpu:
                break;
            case decltype(_target)::Nvidia: {
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
