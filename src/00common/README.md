# 共享库

提供一些基础的类型和函数定义。包括：

| 头文件 | 功能
|:------|:-
| [bf16_t.h](include/common/bf16_t.h)               | 定义 bf16 结构体。
| [data_type.h](include/common/data_type.h)         | 定义 DataType 类型，包含支持的数据类型枚举、与类型关键字的对应关系、计算大小等功能。
| [error_handler.h](include/common/error_handler.h) | 定义异常类型，以及构造各类异常信息的宏。
| [fp16_t.h](include/common/fp16_t.h)               | 定义 fp16 结构体（half，符合 IEEE754 的 16 位浮点数类型）。
| [natural.h](include/common/natural.h)             | 定义自然数迭代器，支持生成从指定整数开始递增的数字。
| [range.h](include/common/range.h)                 | 定义自然数范围，表示指定开始结束的一串自然数并提供范围上的自然数迭代器。
| [rc.h](include/common/rc.h)                       | 定义非原子的引用计数智能指针。
| [slice.h](include/common/slice.h)                 | 提供类似 `std::range` 的功能，用开始和结束指针指示一段连续内存并提供指针作为迭代器。
