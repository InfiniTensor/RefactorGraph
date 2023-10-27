#include "mem_manager/mem_offset_calculator.h"
#include "common.h"
#include <utility>

namespace refactor::mem_manager {

    OffsetCalculator::OffsetCalculator(size_t alignment)
        : _alignment(alignment) {}

    void OffsetCalculator::init() {
        _used = 0;
        _peak = 0;
        _freeBlocks.clear();
        _headAddrToBlockSize.clear();
        _tailAddrToBlockSize.clear();
    }

    size_t OffsetCalculator::alloc(size_t size) {
        // pad the size to the multiple of alignment
        size = getAlignedSize(size);
        auto it = _freeBlocks.lower_bound(FreeBlockInfo{(size_t) 0, size});

        size_t retAddr = _peak;
        if (it != _freeBlocks.end()) {
            // found an alvailable free memory block for allocation
            size_t blockSize = it->blockSize;
            retAddr = it->addr;
            size_t tailAddr = retAddr + size;
            // update the map of head and tail address offset of memory blocks
            _headAddrToBlockSize.erase(retAddr);
            _tailAddrToBlockSize.erase(tailAddr);
            // memory block splitting
            if (blockSize > tailAddr - retAddr) {
                FreeBlockInfo newBlock = {tailAddr,
                                          blockSize - (tailAddr - retAddr)};
                _headAddrToBlockSize[tailAddr] = newBlock.blockSize;
                _tailAddrToBlockSize[retAddr + blockSize] = newBlock.blockSize;
                _freeBlocks.insert(newBlock);
            }
            // update the free balanced tree
            _freeBlocks.erase(it);
            _used += tailAddr - retAddr;
        } else {
            // the allocated memory space is not sufficient for reallocation, it
            // needs to be extended
            auto blockTailWithPeak = _tailAddrToBlockSize.find(_peak);
            if (blockTailWithPeak != _tailAddrToBlockSize.end()) {
                // there is a free block located at the end of the currently
                // allocated memory, where this free block has its tail address as
                // 'peak'
                ASSERT(blockTailWithPeak->second <= _peak,
                       "the free block's size should less or equal than peak");
                retAddr = _peak - blockTailWithPeak->second;
                ASSERT(blockTailWithPeak->second < size,
                       "the available free block's size should less than size");
                _peak += (size - blockTailWithPeak->second);
                // updata _freeBlocks, _headAddrToBlockSize and _tailAddrToBlockSize
                FreeBlockInfo endBlock = {retAddr, blockTailWithPeak->second};
                _freeBlocks.erase(endBlock);
                _headAddrToBlockSize.erase(endBlock.addr);
                _tailAddrToBlockSize.erase(endBlock.addr + endBlock.blockSize);
            } else {
                _peak = _peak + size;
            }
            _used += size;
        }

        return retAddr;
    }

    void OffsetCalculator::free(size_t addr, size_t size) {
        size = getAlignedSize(size);
        auto tailAddr = addr + size;
        FreeBlockInfo block = {addr, tailAddr - addr};
        _headAddrToBlockSize[addr] = block.blockSize;
        _tailAddrToBlockSize[tailAddr] = block.blockSize;
        auto preFreeBlockIter = _tailAddrToBlockSize.find(addr);
        auto subFreeBlockIter = _headAddrToBlockSize.find(tailAddr);
        if (preFreeBlockIter != _tailAddrToBlockSize.end()) {
            // the head address of the memory block to be freed matches the end of a
            // free block, merge them together
            size_t preBlockSize = preFreeBlockIter->second;
            _headAddrToBlockSize.erase(block.addr);
            _headAddrToBlockSize[block.addr - preBlockSize] += block.blockSize;
            _tailAddrToBlockSize.erase(block.addr);
            _tailAddrToBlockSize[tailAddr] += preBlockSize;
            block.addr -= preBlockSize;
            block.blockSize += preBlockSize;
            // delete the preceding adjacent free block
            _freeBlocks.erase(FreeBlockInfo{block.addr, preBlockSize});
        }
        if (subFreeBlockIter != _headAddrToBlockSize.end()) {
            // the tail address of the memory block to be freed matches the start of
            // a free block, merge them together
            auto subBlockSize = subFreeBlockIter->second;
            _headAddrToBlockSize.erase(tailAddr);
            _headAddrToBlockSize[block.addr] += subBlockSize;
            _tailAddrToBlockSize.erase(tailAddr);
            _tailAddrToBlockSize[tailAddr + subBlockSize] += block.blockSize;
            tailAddr += subBlockSize;
            block.blockSize += subBlockSize;
            // delete the succeeding adjacent memory block
            _freeBlocks.erase(
                FreeBlockInfo{tailAddr - subBlockSize, subBlockSize});
        }
        _freeBlocks.insert(block);
        _used -= size;
    }

    size_t OffsetCalculator::peak() {
        return _peak;
    }

    std::string OffsetCalculator::info() {
        std::string infoStr = "Used memory: " +
                              std::to_string(_used) + ", peak memory: " +
                              std::to_string(_peak) + "/n";
        return infoStr;
    }

    size_t OffsetCalculator::getAlignedSize(size_t size) {
        return ((size - 1) / _alignment + 1) * _alignment;
    }

}// namespace refactor::mem_manager
