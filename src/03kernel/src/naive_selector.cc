#include "kernel/naive_selector.h"

namespace refactor::kernel {

    std::unique_ptr<Kernel>
    NaiveSelector::select(std::vector<std::unique_ptr<Candidate>> const &candidates,
                          TensorRefs inputs,
                          TensorRefs outputs) const {
        for (auto const &candidate : candidates) {
            if (candidate->compatible(inputs, outputs)) {
                return candidate->lower();
            }
        }
        return nullptr;
    }

}// namespace refactor::kernel
