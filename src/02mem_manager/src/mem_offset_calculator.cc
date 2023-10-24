#include "mem_manager/mem_offset_calculator.h"
#include "common.h"
#include <utility>

namespace refactor::mem_manager {

    // In
    // cuda-c-programming-guide(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses):
    // Any address of a variable residing in global memory or returned by one of the
    // memory allocation routines from the driver or runtime API is always aligned
    // to at least 256 bytes.
    constexpr size_t alignmentInBytesForCUDA = 256;

    OffsetCalculator::OffsetCalculator(size_t alignment) : alignment(alignment) {
    }

    OffsetCalculator::~OffsetCalculator() {
    }

    void OffsetCalculator::init() {
        used = 0;
        peak = 0;
        freeBlocks.clear();
        headAddrToBlockSize.clear();
        tailAddrToBlockSize.clear();
    }

    size_t OffsetCalculator::alloc(size_t size) {
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);
        auto it = this->freeBlocks.lower_bound(freeBlockInfo{(size_t) 0, size});

        size_t retAddr = this->peak;
        if (it != this->freeBlocks.end()) {
            // found an alvailable free memory block for allocation
            size_t blockSize = it->blockSize;
            retAddr = it->addr;
            size_t tailAddr = retAddr + size;
            // update the map of head and tail address offset of memory blocks
            this->headAddrToBlockSize.erase(retAddr);
            this->tailAddrToBlockSize.erase(tailAddr);
            // memory block splitting
            if (blockSize > tailAddr - retAddr) {
                freeBlockInfo newBlock = {tailAddr,
                                          blockSize - (tailAddr - retAddr)};
                this->headAddrToBlockSize[tailAddr] = newBlock.blockSize;
                this->tailAddrToBlockSize[retAddr + blockSize] = newBlock.blockSize;
                this->freeBlocks.insert(newBlock);
            }
            // update the free balanced tree
            this->freeBlocks.erase(it);
            this->used += tailAddr - retAddr;
        } else {
            // the allocated memory space is not sufficient for reallocation, it
            // needs to be extended
            auto blockTailWithPeak = this->tailAddrToBlockSize.find(this->peak);
            if (blockTailWithPeak != this->tailAddrToBlockSize.end()) {
                // there is a free block located at the end of the currently
                // allocated memory, where this free block has its tail address as
                // 'peak'
                ASSERT(blockTailWithPeak->second < this->peak,
                       "blockTailWithPeak->second should less than this->peak");
                retAddr = this->peak - blockTailWithPeak->second;
                ASSERT(blockTailWithPeak->second < size,
                       "blockTailWithPeak->second should less than size");
                this->peak += (size - blockTailWithPeak->second);
                // updata freeBlocks, headAddrToBlockSize and tailAddrToBlockSize
                freeBlockInfo endBlock = {retAddr, blockTailWithPeak->second};
                this->freeBlocks.erase(endBlock);
                this->headAddrToBlockSize.erase(endBlock.addr);
                this->tailAddrToBlockSize.erase(endBlock.addr + endBlock.blockSize);
            } else {
                this->peak = this->peak + size;
            }
            this->used += size;
        }

        return retAddr;
    }

    void OffsetCalculator::free(size_t addr, size_t size) {
        size = getAlignedSize(size);
        auto tailAddr = addr + size;
        freeBlockInfo block = {addr, tailAddr - addr};
        this->headAddrToBlockSize[addr] = block.blockSize;
        this->tailAddrToBlockSize[tailAddr] = block.blockSize;
        auto preFreeBlockIter = this->tailAddrToBlockSize.find(addr);
        auto subFreeBlockIter = this->headAddrToBlockSize.find(tailAddr);
        if (preFreeBlockIter != this->tailAddrToBlockSize.end()) {
            // the head address of the memory block to be freed matches the end of a
            // free block, merge them together
            size_t preBlockSize = preFreeBlockIter->second;
            this->headAddrToBlockSize.erase(block.addr);
            this->headAddrToBlockSize[block.addr - preBlockSize] += block.blockSize;
            this->tailAddrToBlockSize.erase(block.addr);
            this->tailAddrToBlockSize[tailAddr] += preBlockSize;
            block.addr -= preBlockSize;
            block.blockSize += preBlockSize;
            // delete the preceding adjacent free block
            this->freeBlocks.erase(freeBlockInfo{block.addr, preBlockSize});
        }
        if (subFreeBlockIter != this->headAddrToBlockSize.end()) {
            // the tail address of the memory block to be freed matches the start of
            // a free block, merge them together
            auto subBlockSize = subFreeBlockIter->second;
            this->headAddrToBlockSize.erase(tailAddr);
            this->headAddrToBlockSize[block.addr] += subBlockSize;
            this->tailAddrToBlockSize.erase(tailAddr);
            this->tailAddrToBlockSize[tailAddr + subBlockSize] += block.blockSize;
            tailAddr += subBlockSize;
            block.blockSize += subBlockSize;
            // delete the succeeding adjacent memory block
            this->freeBlocks.erase(
                freeBlockInfo{tailAddr - subBlockSize, subBlockSize});
        }
        this->freeBlocks.insert(block);
        this->used -= size;
    }

    size_t OffsetCalculator::getPeek() {
        return this->peak;
    }

    std::string OffsetCalculator::info() {
        std::string infoStr = "Used memory: " +
                              std::to_string(this->used) + ", peak memory: " +
                              std::to_string(this->peak) + "/n";
    }

    size_t OffsetCalculator::getAlignedSize(size_t size) {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

}// namespace refactor::mem_manager
