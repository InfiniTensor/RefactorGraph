#ifndef RUNTIME_MEM_MANAGER_HH
#define RUNTIME_MEM_MANAGER_HH

#include "common.h"
#include "mem_manager/mem_manager.hh"
#include "resource.h"

namespace refactor::runtime {

    struct MemManager final : public Resource {
        Arc<mem_manager::MemManager> manager;

        static size_t typeId() noexcept;
        static runtime::ResourceBox build(Arc<mem_manager::MemManager>) noexcept;

        size_t resourceTypeId() const noexcept final;
        std::string_view description() const noexcept final;
    };

}// namespace refactor::runtime

#endif// RUNTIME_MEM_MANAGER_HH
