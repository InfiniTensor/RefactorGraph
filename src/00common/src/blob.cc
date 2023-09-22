#include "common/blob.h"
#include <cstdint>

namespace refactor::common {

    Blob::Blob(size_t byteSize) : ptr(new uint8_t[byteSize]) {}
    Blob::~Blob() { delete[] ptr; }

}// namespace refactor::common
