#include "computation/operators/slice.h"

namespace refactor::computation {

    size_t Slice::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Slice::opTypeId() const noexcept { return typeId(); }
    std::string_view Slice::name() const noexcept { return "Slice"; }

}// namespace refactor::computation
