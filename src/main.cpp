#include "common/data_type.h"
#include "common/error_handler.h"
#include "fmtlog.h"

using namespace refactor;

int main() {
    FMTLOG(fmtlog::INF, "The answer is {}.", 42);
    ASSERT(common::parseDataType(3) == common::DataType::I8, "parseDataType failed");
    return 0;
}
