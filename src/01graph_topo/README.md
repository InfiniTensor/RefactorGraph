# 图拓扑

图拓扑结构的增、改、查，包含且仅包含图的拓扑结构。

核心结构通过 [*include/graph_topo/container.hpp*](include/graph_topo/container.hpp) 中的 `GraphTopo` 模板类构造和存储。

由于拓扑结构内部不保存所有连接关系，要快速查询连接关系就需要通过其他结构的缓存。[*include/graph_topo/searcher.hpp*](include/graph_topo/searcher.hpp) 中定义的 `Searcher` 结构可以缓存图的全局输入输出、每个节点的输入输出、节点和节点之间的前驱后继关系等信息，当 `GraphTopo` 构造完毕，可以利用它构造一个 *Searcher* 对象，在访问拓扑结构时也直接通过 *Searcher* 访问。
