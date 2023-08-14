#include "data_type.h"
#include "error_handler.h"

using namespace refactor;

int main() {
    ASSERT(common::parseDataType(3) == common::DataType::I8, "parseDataType failed");
    return 0;
}
