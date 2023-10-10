#include "kernel/collectors/pool_attributes.hh"
#include "common/error_handler.h"
#include "common/range.h"

namespace refactor::kernel {

    PoolAttributes::PoolAttributes(
        size_t rank,
        int64_t const *dilations,
        int64_t const *pads,
        int64_t const *strides)
        : _values((1 + 1 + 2) * rank, 0) {
        constexpr static int64_t LIMIT = std::numeric_limits<uint16_t>::max();

        uint16_t *dilations_ = _values.data(),
                 *strides_ = dilations_ + rank,
                 *pads_ = dilations_ + rank;
        auto range = common::range0_(rank);
        if (dilations) {
            for (auto i : range) {
                auto dilation = dilations[i];
                ASSERT(0 < dilation && dilation <= LIMIT, "dilation out of range");
                dilations_[i] = static_cast<uint16_t>(dilation);
            }
        }
        if (strides) {
            for (auto i : range) {
                auto stride = strides[i];
                ASSERT(0 < stride && stride <= LIMIT, "stride out of range");
                strides_[i] = static_cast<uint16_t>(stride);
            }
        }
        if (pads) {
            for (auto i : range) {
                auto begin = pads[i], end = (pads + rank)[i];

                ASSERT(0 < begin && begin <= LIMIT, "pad out of range");
                pads_[i] = static_cast<uint16_t>(begin);

                ASSERT(0 < end && end <= LIMIT, "pad out of range");
                (pads_ + rank)[i] = static_cast<uint16_t>(end);
            }
        } else {
            std::memset(pads_, 0, rank * 2 * sizeof(uint16_t));
        }
    }

    auto PoolAttributes::rank() const noexcept -> size_t {
        return _values.size() / 4;
    }
    auto PoolAttributes::dilations() const noexcept -> uint16_t const * {
        return _values.data();
    }
    auto PoolAttributes::pads() const noexcept -> uint16_t const * {
        return _values.data() + rank() * 2;
    }
    auto PoolAttributes::padsBegin() const noexcept -> uint16_t const * {
        return _values.data() + rank() * 2;
    }
    auto PoolAttributes::padsEnd() const noexcept -> uint16_t const * {
        return _values.data() + rank() * 3;
    }
    auto PoolAttributes::strides() const noexcept -> uint16_t const * {
        return _values.data() + rank();
    }

}// namespace refactor::kernel
