#include "data_type.h"
#include "kernel_list.h"
#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace proj_namespace;

int main() {
    core::trySubProject();
    assert(core::parseDataType(3) == core::DataType::I8);

    kernel_list::trySubProject();
    return 0;
}
