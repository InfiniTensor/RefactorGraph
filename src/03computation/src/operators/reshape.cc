#include "computation/operators/reshape.h"

namespace refactor::computation {

    size_t Reshape::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Reshape::opTypeId() const {
        return typeId();
    }
    std::string_view Reshape::name() const {
        return "Reshape";
    }
    bool Reshape::isLayoutDependent() const {
        return true;
    }

}// namespace refactor::computation
