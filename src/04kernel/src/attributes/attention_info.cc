#include "kernel/attributes/attention_info.h"

namespace refactor::kernel {

    dim_t AttentionInfo::attLen(dim_t pastSeqLen) const noexcept {
        return pastSeqLen + seqLen;
    }

    size_t AttentionInfo::attSize(dim_t pastSeqLen) const noexcept {
        return batch * nHead * seqLen * attLen(pastSeqLen) * dataType.size();
    }

}// namespace refactor::kernel
