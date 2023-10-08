#include "computation/operators/concat.h"

namespace refactor::computation {

    size_t Concat::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Concat::opTypeId() const noexcept { return typeId(); }
    std::string_view Concat::name() const noexcept { return "Concat"; }

}// namespace refactor::computation
