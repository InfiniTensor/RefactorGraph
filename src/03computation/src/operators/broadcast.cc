#include "computation/operators/broadcast.h"

namespace refactor::computation {

    size_t Broadcast::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Broadcast::opTypeId() const { return typeId(); }
    std::string_view Broadcast::name() const { return "Broadcast"; }
    bool Broadcast::isLayoutDependent() const { return true; }

}// namespace refactor::computation
