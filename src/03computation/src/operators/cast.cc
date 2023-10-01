#include "computation/operators/cast.h"

namespace refactor::computation {

    size_t Cast::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Cast::opTypeId() const { return typeId(); }
    std::string_view Cast::name() const { return "Cast"; }

}// namespace refactor::computation
