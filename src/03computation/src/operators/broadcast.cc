#include "computation/operators/broadcast.h"

namespace refactor::computation {

    size_t Broadcast::typeId() noexcept {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Broadcast::opTypeId() const noexcept { return typeId(); }
    std::string_view Broadcast::name() const noexcept { return "Broadcast"; }

}// namespace refactor::computation
