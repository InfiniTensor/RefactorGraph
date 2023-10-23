#include "mem_manager/foreign_blob.hh"
#include <utility>

namespace refactor::mem_manager {

    ForeignBlob::ForeignBlob(Arc<MemManager> m, size_t bytes)
        : _memManager(std::move(m)),
          _ptr(_memManager->malloc(bytes)) {}
    ForeignBlob::~ForeignBlob() {
        _memManager->free(std::exchange(_ptr, nullptr));
    }

    Arc<ForeignBlob>
    ForeignBlob::share(Arc<MemManager> m, size_t bytes) {
        return Arc<ForeignBlob>(new ForeignBlob(std::move(m), bytes));
    }
    ForeignBlob::operator void const *() const noexcept { return _ptr; }
    ForeignBlob::operator void *() noexcept { return _ptr; }
    void ForeignBlob::copyIn(void const *host, size_t bytes) {
        _memManager->copyHD(_ptr, host, bytes);
    }
    void ForeignBlob::copyOut(void *host, size_t bytes) const {
        _memManager->copyDH(host, _ptr, bytes);
    }
    void ForeignBlob::copyFrom(ForeignBlob const &src, size_t bytes) {
        _memManager->copyDD(_ptr, src._ptr, bytes);
    }
    void ForeignBlob::copyTo(ForeignBlob &tgt, size_t bytes) const {
        _memManager->copyDD(tgt._ptr, _ptr, bytes);
    }
}// namespace refactor::mem_manager
