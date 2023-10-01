#include "computation/operators/conv.h"

namespace refactor::computation {

    size_t Conv::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Conv::opTypeId() const { return typeId(); }
    std::string_view Conv::name() const { return "Conv"; }

}// namespace refactor::computation
