#ifndef MEM_MANAGER_MEM_OFFSET_CALCULATOR_H
#define MEM_MANAGER_MEM_OFFSET_CALCULATOR_H

#include <cstddef>
#include <set>
#include <string>
#include <unordered_map>

namespace refactor::mem_manager {

    class OffsetCalculator {
    private:
        size_t _used = 0;

        size_t _peak = 0;

        size_t _alignment;

        struct FreeBlockInfo {
            size_t addr;
            size_t blockSize;
        };

        struct CmpFreeBlockInfo {
            bool operator()(const FreeBlockInfo &a, const FreeBlockInfo &b) const {
                return (a.blockSize != b.blockSize) ? (a.blockSize < b.blockSize)
                                                    : (a.addr < b.addr);
            }
        };

        // free balanced tree, maintains all free memory blocks
        std::set<FreeBlockInfo, CmpFreeBlockInfo> _freeBlocks;

        // key: head address offset of the free memory block
        // value: blockSize of the block
        std::unordered_map<size_t, size_t> _headAddrToBlockSize;

        // key: tail address offset of the free memory block
        // value: blockSize of the block
        std::unordered_map<size_t, size_t> _tailAddrToBlockSize;

    public:
        explicit OffsetCalculator(size_t alignment);
        virtual ~OffsetCalculator() = default;

        void init();

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

        size_t peak();

        std::string info();

    private:
        // function: memory alignment, rouned up
        // return: size of the aligned memory block
        size_t getAlignedSize(size_t size);
    };

}// namespace refactor::mem_manager

#endif// MEM_MANAGER_MEM_OFFSET_CALCULATOR_H
