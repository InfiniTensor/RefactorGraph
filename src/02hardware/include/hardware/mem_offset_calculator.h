#ifndef MEM_MANAGER_MEM_OFFSET_CALCULATOR_H
#define MEM_MANAGER_MEM_OFFSET_CALCULATOR_H

#include <optional>
#include <set>
#include <string>
#include <unordered_map>

namespace refactor::hardware {

    class OffsetCalculator {
        size_t
            _used,
            _peak,
            _alignment;

        struct FreeBlockInfo {
            size_t addr, blockSize;

            bool operator<(FreeBlockInfo const &) const noexcept;
        };

        // free balanced tree, maintains all free memory blocks
        std::set<FreeBlockInfo> _freeBlocks;

        // key: head address offset of the free memory block
        // value: blockSize of the block
        std::unordered_map<size_t, size_t> _headAddrToBlockSize;

        // key: tail address offset of the free memory block
        // value: blockSize of the block
        std::unordered_map<size_t, size_t> _tailAddrToBlockSize;

        struct TraceInfo {
            size_t allocTimes, freeTimes;
        };
        std::optional<TraceInfo> _traceInfo;
        // trace format :
        // CALCULATOR this alloc/free begin end size allocTimes freeTimes peak rate freeCount minBlockSize maxBlockSize
        void trace(std::string event);

    public:
        explicit OffsetCalculator(size_t alignment, bool trace = false);

        // function: simulate memory allocation
        // arguments:
        //     size: size of memory block to be allocated
        // return: head address offset of the allocated memory block
        size_t alloc(size_t size);

        // function: simulate memory free
        // arguments:
        //     addr: head address offset of memory block to be free
        //     size: size of memory block to be freed
        void free(size_t addr, size_t size);

        size_t peak() const noexcept;
    };

}// namespace refactor::hardware

#endif// MEM_MANAGER_MEM_OFFSET_CALCULATOR_H
