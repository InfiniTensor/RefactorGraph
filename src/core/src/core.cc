#include "core.h"
#include <fmt/core.h>

namespace proj_namespace::core {
    void try_sub_project() {
        fmt::print("This is 'core' lib.\n");
    }

    std::optional<DataType> parse_data_type(uint8_t value) {
        switch (value) {
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
            case 9:
            case 10:
            case 11:
            case 12:
            case 13:
                return {(DataType) value};
            default:
                return {};
        }
    }
}// namespace proj_namespace::core
