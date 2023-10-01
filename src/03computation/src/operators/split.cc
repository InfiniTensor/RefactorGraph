#include "computation/operators/split.h"

namespace refactor::computation {

    size_t Split::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Split::opTypeId() const { return typeId(); }
    std::string_view Split::name() const { return "Split"; }

}// namespace refactor::computation
