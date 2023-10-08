#include "computation/operators/reshape.h"

namespace refactor::computation {

    size_t Reshape::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Reshape::opTypeId() const noexcept { return typeId(); }
    std::string_view Reshape::name() const noexcept { return "Reshape"; }

}// namespace refactor::computation
