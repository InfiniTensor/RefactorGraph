#include "computation/operators/mat_mul.h"

namespace refactor::computation {

    size_t MatMul::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t MatMul::opTypeId() const { return typeId(); }
    std::string_view MatMul::name() const { return "MatMul"; }
    bool MatMul::isLayoutDependent() const { return true; }

}// namespace refactor::computation
