#ifndef MEM_MANAGER_MEM_OFFSET_CALCULATOR_H
#define MEM_MANAGER_MEM_OFFSET_CALCULATOR_H

#include <cstddef>
#include <set>
#include <string>
#include <unordered_map>

namespace refactor::mem_manager {

    class OffsetCalculator {
    private:
        size_t used = 0;

        size_t peak = 0;

        size_t alignment;

        struct freeBlockInfo {
            size_t addr;
            size_t blockSize;
        };

        struct cmpFreeBlockInfo {
            bool operator()(const freeBlockInfo &a, const freeBlockInfo &b) const {
                return (a.blockSize != b.blockSize) ? (a.blockSize < b.blockSize)
                                                    : (a.addr < b.addr);
            }
        };

        // free balanced tree, maintains all free memory blocks
        std::set<freeBlockInfo, cmpFreeBlockInfo> freeBlocks;

        // key: head address offset of the free memory block
        // value: blockSize of the block
        std::unordered_map<size_t, size_t> headAddrToBlockSize;

        // key: tail address offset of the free memory block
        // value: blockSize of the block
        std::unordered_map<size_t, size_t> tailAddrToBlockSize;

    public:
        OffsetCalculator(size_t alignment);

        virtual ~OffsetCalculator();

        void init();

        // function: simulate memory allocation
        // arguments：
        //     size: size of memory block to be allocated
        // return: head address offset of the allocated memory block
        size_t alloc(size_t size);

        // function: simulate memory free
        // arguments:
        //     addr: head address offset of memory block to be free
        //     size: size of memory block to be freed
        void free(size_t addr, size_t size);

        size_t getPeek();

        std::string info();

    private:
        // function: memory alignment, rouned up
        // return: size of the aligned memory block
        size_t getAlignedSize(size_t size);
    };

}// namespace refactor::mem_manager

#endif// MEM_MANAGER_MEM_OFFSET_CALCULATOR_H
