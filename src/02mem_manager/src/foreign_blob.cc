#include "mem_manager/foreign_blob.hh"
#include <utility>

namespace refactor::mem_manager {

    ForeignBlob::ForeignBlob(MemFunctions f, size_t bytes)
        : _memFunctions(f),
          _ptr(f.malloc(bytes)) {}
    ForeignBlob::~ForeignBlob() {
        _memFunctions.free(std::exchange(_ptr, nullptr));
    }

    std::shared_ptr<ForeignBlob>
    ForeignBlob::share(MemFunctions f, size_t bytes) {
        return std::shared_ptr<ForeignBlob>(new ForeignBlob(f, bytes));
    }
    ForeignBlob::operator void const *() const noexcept { return _ptr; }
    ForeignBlob::operator void *() noexcept { return _ptr; }
    void ForeignBlob::copyIn(void const *host, size_t bytes) {
        _memFunctions.copyHd(_ptr, host, bytes);
    }
    void ForeignBlob::copyOut(void *host, size_t bytes) const {
        _memFunctions.copyDh(host, _ptr, bytes);
    }
    void ForeignBlob::copyFrom(ForeignBlob const &src, size_t bytes) {
        _memFunctions.copyDd(_ptr, src._ptr, bytes);
    }
    void ForeignBlob::copyTo(ForeignBlob &tgt, size_t bytes) const {
        _memFunctions.copyDd(tgt._ptr, _ptr, bytes);
    }
}// namespace refactor::mem_manager
