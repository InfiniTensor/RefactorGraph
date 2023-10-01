#include "computation/operators/where.h"

namespace refactor::computation {

    size_t Where::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Where::opTypeId() const { return typeId(); }
    std::string_view Where::name() const { return "Where"; }

}// namespace refactor::computation
