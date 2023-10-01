#include "computation/operators/identity.h"

namespace refactor::computation {

    size_t Identity::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Identity::opTypeId() const { return typeId(); }
    std::string_view Identity::name() const { return "Identity"; }

}// namespace refactor::computation
