#include "computation/operators/softmax.h"

namespace refactor::computation {

    size_t Softmax::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Softmax::opTypeId() const { return typeId(); }
    std::string_view Softmax::name() const { return "Softmax"; }

}// namespace refactor::computation
