#include "kernel/attributes/expand_info.h"
#include <numeric>

namespace refactor::kernel {

    bool ExpandInfo::Dim::operator==(Dim const &rhs) const noexcept {
        return i == rhs.i && o == rhs.o;
    }
    bool ExpandInfo::Dim::operator!=(Dim const &rhs) const noexcept {
        return !operator==(rhs);
    }

    ExpandInfo::ExpandInfo(
        std::vector<Dim> strides_,
        dim_t blockCount_,
        dim_t blockSize_) noexcept
        : strides(std::move(strides_)),
          blockCount(blockCount_),
          blockSize(blockSize_) {}

    ExpandInfo::ExpandInfo(
        Tensor const &input,
        Tensor const &output) noexcept
        : strides{{1, 1}},
          blockCount(1),
          blockSize(input.dataType.size()) {
        ASSERT(input.rank() <= output.rank(), "Unreachable");
        auto i = input.shape.rbegin(),
             ei = input.shape.rend(),
             o = output.shape.rbegin(),
             eo = output.shape.rend();
        dim_t stride = 1;
        while (o != eo) {
            auto i_ = i == ei ? 1 : *i++,
                 o_ = *o++;
            if (o_ == 1) { continue; }
            if (auto &it = strides.back(); i_ == 1) {
                if (it.i != 0) {
                    strides.push_back({0, blockCount});
                }
            } else {
                if (it.i == 0) {
                    strides.push_back({stride, blockCount});
                }
                stride *= i_;
            }
            blockCount *= o_;
        }
        if (strides.size() == 1) {
            // 没有发生广播
            blockSize *= blockCount;
            blockCount = 1;
            strides = {};
            return;
        }
        std::reverse(strides.begin(), strides.end());
        strides.pop_back();

        auto tail = strides.back();
        ASSERT(tail.i == 0, "Unreachable");

        blockSize *= tail.o;
        blockCount /= tail.o;
        for (auto &s : strides) {
            s.i /= tail.o;
            s.o /= tail.o;
        }
    }

    ExpandInfo ExpandInfo::reform(dim_t maxblockSize) const noexcept {
        auto ans = *this;
        ans.reformAssign(maxblockSize);
        return ans;
    }
    void ExpandInfo::reformAssign(dim_t maxblockSize) noexcept {
        auto blockSize_ = std::gcd(blockSize, maxblockSize);
        if (blockSize_ == blockSize) { return; }
        auto times = blockSize / blockSize_;
        blockCount *= times;
        blockSize = blockSize_;
        if (!strides.empty()) {
            for (auto &s : strides) {
                s.i *= times;
                s.o *= times;
            }
            strides.resize(strides.size() + 1);
            strides.back() = {1, 1};
        }
    }

}// namespace refactor::kernel
