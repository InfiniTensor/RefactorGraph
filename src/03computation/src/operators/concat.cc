#include "computation/operators/concat.h"

namespace refactor::computation {

    size_t Concat::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t Concat::opTypeId() const {
        return typeId();
    }
    std::string_view Concat::name() const {
        return "Concat";
    }

}// namespace refactor::computation
