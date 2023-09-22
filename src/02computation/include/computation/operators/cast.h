#ifndef COMPUTATION_CAST_H
#define COMPUTATION_CAST_H

#include "../operator.h"
#include "common/data_type.h"

namespace refactor::computation {

    struct Cast : public Operator {
        common::DataType targetDataType;
    };

}// namespace refactor::computation

#endif// COMPUTATION_CAST_H
