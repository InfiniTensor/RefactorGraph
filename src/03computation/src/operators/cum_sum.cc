#include "computation/operators/cum_sum.h"

namespace refactor::computation {

    size_t CumSum::typeId() {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    size_t CumSum::opTypeId() const { return typeId(); }
    std::string_view CumSum::name() const { return "CumSum"; }

}// namespace refactor::computation
