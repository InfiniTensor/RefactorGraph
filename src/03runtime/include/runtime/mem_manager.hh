#ifndef RUNTIME_MEM_MANAGER_HH
#define RUNTIME_MEM_MANAGER_HH

#include "common.h"
#include "hardware/mem_manager.hh"
#include "resource.h"

namespace refactor::runtime {

    struct MemManager final : public Resource {
        Arc<hardware::MemManager> manager;

        static size_t typeId() noexcept;
        static runtime::ResourceBox build(Arc<hardware::MemManager>) noexcept;

        size_t resourceTypeId() const noexcept final;
        std::string_view description() const noexcept final;
    };

}// namespace refactor::runtime

#endif// RUNTIME_MEM_MANAGER_HH
