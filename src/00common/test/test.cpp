#include "refactor/common.h"
#include <cassert>
#include <fmt/core.h>

using namespace refactor;

int main() {
    float val = 2047;
    fp16_t ans(val);
    assert(ans.to_f32() == val);
    fmt::println("{}", ans.to_string());
    return 0;
}
