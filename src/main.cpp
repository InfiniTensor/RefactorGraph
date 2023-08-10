#include "data_type.h"
#include "graph_topo.hpp"
#include "kernel_list.h"
#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace refactor;

int main() {
    common::trySubProject();
    assert(common::parseDataType(3) == common::DataType::I8);

    kernel_list::trySubProject();
    return 0;
}
