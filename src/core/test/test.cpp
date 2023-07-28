#include "fp16.h"
#include <cassert>
#include <fmt/core.h>

int main() {
    fp16_t val(192.0f);
    assert(val.to_f32() == 192.0f);
    fmt::println("{}", val.to_string());
    return 0;
}
