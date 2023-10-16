#include "computation/operators/gather_elements.h"

namespace refactor::computation {

    size_t GatherElements::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t GatherElements::opTypeId() const noexcept { return typeId(); }
    std::string_view GatherElements::name() const noexcept { return "GatherElements"; }

}// namespace refactor::computation
