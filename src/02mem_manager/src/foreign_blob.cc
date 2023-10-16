#include "mem_manager/foreign_blob.hh"
#include <utility>

namespace refactor::mem_manager {

    ForeignBlob::ForeignBlob(MemFunctions const &f, size_t bytes)
        : _memFunctions(f),
          _ptr(f.malloc(bytes)) {}
    ForeignBlob::~ForeignBlob() {
        _memFunctions.free(std::exchange(_ptr, nullptr));
    }

    std::shared_ptr<ForeignBlob>
    ForeignBlob::share(MemFunctions const &f, size_t bytes) {
        return std::shared_ptr<ForeignBlob>(new ForeignBlob(f, bytes));
    }
    ForeignBlob::operator void const *() const noexcept { return _ptr; }
    ForeignBlob::operator void *() noexcept { return _ptr; }
    uint8_t *ForeignBlob::ptr() noexcept {
        return reinterpret_cast<uint8_t *>(_ptr);
    }

}// namespace refactor::mem_manager
