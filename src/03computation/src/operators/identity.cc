#include "computation/operators/identity.h"

namespace refactor::computation {

    size_t Identity::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Identity::opTypeId() const noexcept { return typeId(); }
    std::string_view Identity::name() const noexcept { return "Identity"; }

}// namespace refactor::computation
