#include "computation/operators/gather.h"

namespace refactor::computation {

    size_t Gather::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Gather::opTypeId() const noexcept { return typeId(); }
    std::string_view Gather::name() const noexcept { return "Gather"; }

}// namespace refactor::computation
