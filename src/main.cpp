#include "data_type.h"
#include "kernel_list.h"
#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace proj_namespace;

int main() {
    core::try_sub_project();
    assert(core::parse_data_type(3) == core::DataType::I8);

    kernel_list::try_sub_project();
    return 0;
}
