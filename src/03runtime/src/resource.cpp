#include "runtime/resource.h"
#include "common.h"

namespace refactor::runtime {

    auto Resource::is(size_t id) const noexcept -> bool {
        return resourceTypeId() == id;
    }

    auto Resources::fetch(size_t id) noexcept -> Resource * {
        auto it = _internal.find(id);
        return it != _internal.end() ? it->second.get() : nullptr;
    }
    auto Resources::fetchOrStore(ResourceBox resource) noexcept -> Resource * {
        auto [it, ok] = _internal.try_emplace(resource->resourceTypeId(), std::move(resource));
        return it->second.get();
    }
    auto Resources::fetchOrStore(size_t id, std::function<ResourceBox()> fn) noexcept -> Resource * {
        auto it = _internal.find(id);
        if (it == _internal.end()) {
            std::tie(it, std::ignore) = _internal.insert({id, fn()});
        }
        return it->second.get();
    }

}// namespace refactor::runtime
