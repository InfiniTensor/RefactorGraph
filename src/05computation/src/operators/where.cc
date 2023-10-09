#include "computation/operators/where.h"

namespace refactor::computation {

    size_t Where::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Where::opTypeId() const noexcept { return typeId(); }
    std::string_view Where::name() const noexcept { return "Where"; }

}// namespace refactor::computation
