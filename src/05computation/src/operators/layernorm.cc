#include "computation/operators/layernorm.h"

namespace refactor::computation {
    using Op = LayerNormalization;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "LayerNormalization"; }
    auto Op::serialize() const noexcept -> std::string {
        union code {
            float f;
            int32_t i;
        };
        return fmt::format(("{}({:e}={:#010x},{})"),
                           name(), epsilon,
                           code{epsilon}.i, axis);
    }

}// namespace refactor::computation
