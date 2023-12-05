#ifndef RUNTIME_STREAM_H
#define RUNTIME_STREAM_H

#include "graph_topo.h"
#include "hardware/device.h"
#include "resource.h"
#include <chrono>

namespace refactor::runtime {
    using Routine = std::function<void(runtime::Resources &, void *, void const *const *, void *const *)>;

    void emptyRoutine(runtime::Resources &, void *, void const *const *, void *const *);

    struct Node {
        Routine routine;
        size_t workspaceOffset;

        template<class T>
        Node(T &&r, size_t wso = 0) noexcept
            : routine(std::forward<T>(r)),
              workspaceOffset(wso) {}
    };

    struct Edge {
        Arc<hardware::Device::Blob> blob;
        size_t stackOffset;
    };

    class Stream {
        Arc<hardware::Device> _device;
        Arc<hardware::Device::Blob> _stack;
        Resources _resources;
        graph_topo::Graph<Node, Edge> _graph;

    public:
        Stream(decltype(_resources),
               size_t,
               graph_topo::GraphTopo,
               std::vector<Node>,
               std::vector<Edge>,
               decltype(_device));
        void setData(count_t, void const *, size_t);
        void setData(count_t, Arc<hardware::Device::Blob>);
        bool getData(count_t, void *, size_t) const;
        Resources& getResources();
        void run();
        auto bench(void (*sync)()) -> std::vector<std::chrono::nanoseconds>;
        void trace(std::function<void(count_t, void const *const *, void const *const *)>);
    };

}// namespace refactor::runtime

#endif// RUNTIME_STREAM_H
