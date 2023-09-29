#include "computation/operators/transpose.h"

namespace refactor::computation {

    size_t Transpose::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Transpose::opTypeId() const {
        return typeId();
    }
    std::string_view Transpose::name() const {
        return "Transpose";
    }

}// namespace refactor::computation
