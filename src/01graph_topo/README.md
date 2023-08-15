# 图拓扑

图拓扑结构的增、改、查，包含且仅包含图的拓扑结构。

核心结构通过 [*include/graph_topo.hpp*](include/graph_topo.hpp) 中的 `GraphTopo` 模板类构造和存储。`GraphTopo` 中的三个 `std::vector` 分别保存图中的节点、边和边到节点的连接，其中节点到边的映射为前向星，边到节点的映射为链式前向星。使用这种结构存储的好处有：

1. 构造逻辑简单：每种连接关系只保存一个方向、一次，且所有构造操作都是 `push_back`，不需要中间插入；
2. 对象操作简单：内部结构干净，栈上只有 24x3 = 72 字节，执行拷贝、移动快，结构中只有对象序号，没有指向外部的指针，移动无负担；
3. 内存局部性好：连接关系保存在 3 个连续内存上的 `std::vector`，遍历快；

由于拓扑结构内部不保存所有连接关系，要快速查询连接关系就需要通过其他结构的缓存。[*include/graph_topo_searcher.hpp*](include/graph_topo_searcher.hpp) 中定义的 `GaraphTopoSearcher` 结构可以缓存图的全局输入输出、每个节点的输入输出、节点和节点之间的前驱后继关系等信息，当 `GraphTopo` 构造完毕，可以利用它构造一个 *Searcher* 对象，在访问拓扑结构时也直接通过 *Searcher* 访问。
