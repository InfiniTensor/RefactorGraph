#include "computation/operators/gather.h"

namespace refactor::computation {

    size_t Gather::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Gather::opTypeId() const {
        return typeId();
    }
    std::string_view Gather::name() const {
        return "Gather";
    }

}// namespace refactor::computation
