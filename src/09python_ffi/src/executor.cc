#include "executor.h"
#include <filesystem>
#include <fstream>

#ifdef USE_CUDA
#include "kernel/cuda/functions.cuh"
#endif// USE_CUDA

namespace refactor::python_ffi {

    Executor::Executor(computation::Graph graph, runtime::Stream stream)
        : _graph(std::move(graph)),
          _stream(std::move(stream)) {}

    void Executor::setInput(count_t i, pybind11::array data) {
        _stream.setInput(i, data.data(), data.nbytes());
    }

    auto Executor::getOutput(count_t i) -> pybind11::array {
        auto globalOutputs = _graph.internal().contiguous().topology.globalOutputs();
        ASSERT(i < globalOutputs.size(), "input index out of range");

        auto const &tensor = *_graph.internal().contiguous().edges[globalOutputs[i]].tensor;
        auto ans = pybind11::array(buildNumpyDType(tensor.dataType), std::move(tensor.shape), nullptr);
        _stream.getOutput(i, ans.mutable_data(), ans.nbytes());
        return ans;
    }

    auto Executor::prepare() -> std::vector<count_t> {
        return _stream.prepare();
    }

    void Executor::run() {
        _stream.run();
    }

    void Executor::bench(bool sync) {
#ifdef USE_CUDA
        auto ans = _stream.bench(sync ? kernel::cuda::sync : nullptr);
#else
        auto ans = _stream.bench(nullptr);
#endif// USE_CUDA
        auto const &nodes = _graph.internal().contiguous().nodes;
        for (auto i : range0_(nodes.size())) {
            fmt::println("{} {} {}",
                         i,
                         nodes[i].name,
                         std::chrono::duration_cast<std::chrono::microseconds>(ans[i]).count());
        }
    }

    void Executor::trace(std::string path_) {
        namespace fs = std::filesystem;

        auto path = fs::path(std::move(path_));
        fs::create_directories(path);
        ASSERT(fs::is_directory(path), "Failed to create \"{}\"", path.c_str());

        auto it = _graph.internal().contiguous().topology.begin();
        _stream.trace([&](count_t nodeIdx, void const *const *inputs, void const *const *outputs) {
            auto [nodeIdx_, i_, o_] = *it++;
            ASSERT(nodeIdx_ == nodeIdx, "node index mismatch");
            auto nodeName = _graph.internal().contiguous().nodes[nodeIdx].name;
            std::replace(nodeName.begin(), nodeName.end(), '/', '_');
            std::replace(nodeName.begin(), nodeName.end(), '.', '-');

            std::vector<char> buffer;
            auto const &edges = _graph.internal().contiguous().edges;
            auto fn = [&](char dir, count_t idx, count_t edgeIdx, void const *const *addresses) {
                if (!addresses[idx]) { return; }
                auto const &edge = edges[edgeIdx];

                auto edgeName = edge.name;
                std::replace(edgeName.begin(), edgeName.end(), '/', '_');
                std::replace(edgeName.begin(), edgeName.end(), '.', '-');
                auto file = path / fmt::format("{}({}_{}{}).bin", edgeName, nodeName, dir, idx);
                fs::remove(file);
                std::ofstream os(file, std::ios::binary);

                auto size = edge.tensor->bytesSize();
                buffer.resize(size);
#ifdef USE_CUDA
                kernel::cuda::copyOut(buffer.data(), addresses[idx], size);
#endif
                os.write(buffer.data(), size);
            };

            for (auto i : range0_(i_.size())) { fn('i', i, i_[i], inputs); }
            for (auto i : range0_(o_.size())) { fn('o', i, o_[i], outputs); }
        });
    }

    void Executor::debugInfo() const noexcept {
        auto const &nodes = _graph.internal().contiguous().nodes;
        for (auto i : range0_(nodes.size())) {
            fmt::println("{}. {}", i, nodes[i].name);
        }
    }

}// namespace refactor::python_ffi
