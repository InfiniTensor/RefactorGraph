#include "kernel/attributes/pad_info.h"
#include <numeric>

namespace refactor::kernel {
    using PI = PadInfo;

    // bool PI::Dim::operator==(Dim const &rhs) const noexcept {
    //     return strideI == rhs.strideI &&
    //            strideO == rhs.strideO &&
    //            padStride == rhs.padStride &&
    //            dimt.dimI == rhs.dimI &&;
    // }
    // bool PI::Dim::operator!=(Dim const &rhs) const noexcept {
    //     return !operator==(rhs);
    // }

    PI::PadInfo(decltype(dims) dims_, dim_t blockCount_, dim_t blockSize_) noexcept
        : dims(std::move(dims_)), blockCount(blockCount_), blockSize(blockSize_) {}

    PI::PadInfo(PadDimension dims_, Tensor const &input) : dims{}, blockCount(1),
                                                           blockSize(input.dataType.size()) {
        size_t rank = input.rank();
        ASSERT(dims_.size() == rank, "Invalid to get PadInfo.");

        //	std::vector<dim_t> shape;
        size_t j = 0;
        for (auto i : range0_(rank)) {
            if (dims_[i].dimI != dims_[i].dimO || dims_[i].dimI != 1) {
                if (j < i) { dims_[j] = dims_[i]; }
                //shape.push_back(dims_[i].dimI);
                j++;
            }
        }
        dims_.resize(rank = j);
        // 合并末尾连续维度
        for (auto i : range0_(rank).rev()) {
            if (auto d = dims_[i].dimI; d == dims_[i].dimO) {
                blockSize *= d;
                dims_.pop_back();
            } else {
                dims.reserve(rank = dims_.size());
                auto &dim = dims_[i];
                if (auto times = std::gcd(std::gcd(dims_[i].dimI, dims_[i].pads), dims_[i].dimO); times > 1) {
                    blockSize *= times;
                    dim.dimI /= times;
                    dim.dimO /= times;
                    dim.pads /= times;
                }
                break;
            }
        }

        dim_t strideI = 1, strideO = 1;
        for (auto i : range0_(rank).rev()) {
            auto const &dim = dims_[i];
            dims.push_back({
                strideI,
                strideO,
                static_cast<dim_t>(dim.pads),
                static_cast<dim_t>(dim.dimI),
            });
            strideI *= dim.dimI;
            strideO *= dim.dimO;
        }
        std::reverse(dims.begin(), dims.end());
        // for (auto i : range0_(rank)) {
        //     fmt::println("strideI = {}, strideO = {}, padS = {}, dimI = {}", dims[i].strideI, dims[i].strideO, dims[i].padS, dims[i].dimI);
        // }
        blockCount = strideO;
    }

    void PI::reform(dim_t maxblockSize) noexcept {
        auto blockSize_ = std::gcd(blockSize, maxblockSize);
        if (blockSize_ == blockSize) { return; }
        auto t = blockSize / blockSize_;
        blockCount *= t;
        blockSize = blockSize_;
        for (auto &d : dims) {
            d.strideI *= t;
            d.strideO *= t;
            d.padS *= t;
            d.dimI *= t;
        }
        dims.resize(dims.size() + 1);
        dims.back() = {1, 1, 0, t};
    }

}// namespace refactor::kernel
