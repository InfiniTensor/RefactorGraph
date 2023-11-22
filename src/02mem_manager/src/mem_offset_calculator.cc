#include "mem_manager/mem_offset_calculator.h"
#include "common.h"
#include "mem_manager/functions.h"
#include <utility>

namespace refactor::mem_manager {

    auto OffsetCalculator::FreeBlockInfo::
    operator<(const FreeBlockInfo &rhs) const noexcept -> bool {
        // most likely the two blockSizes are not equal
        return blockSize < rhs.blockSize || (blockSize == rhs.blockSize && addr < rhs.addr);
    }

    OffsetCalculator::OffsetCalculator(size_t alignment, bool trace)
        : _used(0),
          _peak(0),
          _alignment(alignment),
          _freeBlocks{},
          _headAddrToBlockSize{},
          _tailAddrToBlockSize{},
          _trace(trace) {}

    size_t OffsetCalculator::alloc(size_t size) {
        // pad the size to the multiple of alignment
        size = alignBytes(size, _alignment);
        _used += size;

        // found an alvailable free memory block for allocation
        if (auto it = _freeBlocks.lower_bound(FreeBlockInfo{0ul, size}); it != _freeBlocks.end()) {

            auto [addr, blockSize] = *it;
            auto head = addr,
                 tail = head + blockSize;
            // update the map of head and tail address offset of memory blocks
            _freeBlocks.erase(it);
            _headAddrToBlockSize.erase(head);
            _tailAddrToBlockSize.erase(tail);
            // memory block splitting
            if (blockSize > size) {
                auto newSize = blockSize - size;
                head += size;
                _headAddrToBlockSize.emplace(head, newSize);
                _tailAddrToBlockSize.emplace(tail, newSize);
                _freeBlocks.insert({head, newSize});
            }
            if (_trace) {
                fmt::println("alloc {:>10} bytes at offset {:>10}", size, addr);
                fmt::println(info());
            }
            return addr;
        }
        // found an free memory block for allocation located at the end of the currently allocated memory
        if (auto it = _tailAddrToBlockSize.find(_peak); it != _tailAddrToBlockSize.end()) {

            auto blockSize = it->second,
                 addr = _peak - blockSize;
            ASSERT(blockSize < size, "the available free block's size should less than size");
            _freeBlocks.erase({addr, blockSize});
            _headAddrToBlockSize.erase(addr);
            _tailAddrToBlockSize.erase(it);
            _peak += size - blockSize;
            if (_trace) {
                if (size - blockSize > 0) {
                    fmt::println("[Info] Insufficient memory capacity requested at present, increase the peak.");
                }
                fmt::println("alloc {:>10} bytes at offset {:>10}", size, addr);
                fmt::println(info());
            }
            return addr;
        }
        // no available free memory block for allocation, allocate memory at the end of the currently allocated memory
        {
            auto addr = _peak;
            _peak += size;
            if (_trace) {
                fmt::println("[Info] Insufficient memory capacity requested at present, increase the peak.");
                fmt::println("alloc {:>10} bytes at offset {:>10}", size, addr);
                fmt::println(info());
            }
            return addr;
        }
    }

    void OffsetCalculator::free(size_t addr, size_t size) {
        // pad the size to the multiple of alignment
        size = alignBytes(size, _alignment);
        _used -= size;
        if (_trace) {
            fmt::println("free {:>10} bytes at offset {:>10}", size, addr);
            fmt::println(info());
        }

        auto head = addr,
             tail = addr + size;
        if (auto it = _tailAddrToBlockSize.find(head); it != _tailAddrToBlockSize.end()) {
            head -= it->second;
            size += it->second;
            _freeBlocks.erase({head, it->second});
            _headAddrToBlockSize.erase(head);
            _tailAddrToBlockSize.erase(it);
            if (_trace) {
                fmt::println("[Info] coalescence with freeBlock [head: {}, size: {}]", head, it->second);
            }
        }
        if (auto it = _headAddrToBlockSize.find(tail); it != _headAddrToBlockSize.end()) {
            tail += it->second;
            size += it->second;
            _freeBlocks.erase({it->first, it->second});
            _headAddrToBlockSize.erase(it);
            _tailAddrToBlockSize.erase(tail);
            if (_trace) {
                fmt::println("[Info] coalescence with freeBlock [head:{}, size: {}]", it->first, it->second);
            }
        }
        _freeBlocks.insert({head, size});
        _headAddrToBlockSize.emplace(head, size);
        _tailAddrToBlockSize.emplace(tail, size);
    }

    size_t OffsetCalculator::peak() const noexcept {
        return _peak;
    }

    std::string OffsetCalculator::info() const noexcept {
        return fmt::format("[Info] Used memory: {:>10}, peak memory: {:>10}", _used, _peak);
    }

}// namespace refactor::mem_manager
