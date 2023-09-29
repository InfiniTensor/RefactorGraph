#include "computation/operators/gather_elements.h"

namespace refactor::computation {

    size_t GatherElements::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t GatherElements::opTypeId() const {
        return typeId();
    }
    std::string_view GatherElements::name() const {
        return "GatherElements";
    }

}// namespace refactor::computation
