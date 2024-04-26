#include "computation/operators/gelu.h"

namespace refactor::computation {
    using Op = Gelu;

    auto Op::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto Op::opTypeId() const noexcept -> size_t { return typeId(); }
    auto Op::name() const noexcept -> std::string_view { return "Gelu"; }
    auto Op::serialize() const noexcept -> std::string {
        return fmt::format(("{}()"), name());
    }

}// namespace refactor::computation
