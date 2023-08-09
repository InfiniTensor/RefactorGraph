#include "data_type.h"
#include <fmt/core.h>

namespace proj_namespace::core {
    void trySubProject() {
        fmt::println("This is 'core' lib.");
    }

    std::optional<DataType> parseDataType(uint8_t value) {
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
