#include "computation/operators/slice.h"

namespace refactor::computation {

    size_t Slice::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Slice::opTypeId() const { return typeId(); }
    std::string_view Slice::name() const { return "Slice"; }
    bool Slice::isLayoutDependent() const { return true; }

}// namespace refactor::computation
