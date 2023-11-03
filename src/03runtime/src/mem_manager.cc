#include "runtime/mem_manager.hh"

namespace refactor::runtime {

    auto MemManager::typeId() noexcept -> size_t {
        static uint8_t ID = 1;
        return reinterpret_cast<size_t>(&ID);
    }
    auto MemManager::build(Arc<mem_manager::MemManager> ptr) noexcept -> ResourceBox {
        auto ans = std::make_unique<MemManager>();
        ans->manager = std::move(ptr);
        return ans;
    }

    auto MemManager::resourceTypeId() const noexcept -> size_t {
        return typeId();
    }
    auto MemManager::description() const noexcept -> std::string_view {
        return "Memory Manager";
    }

}// namespace refactor::runtime
