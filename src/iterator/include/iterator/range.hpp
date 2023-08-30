#include <functional>
#include <optional>

/// @brief 筛选迭代器。
/// @tparam T 筛选的元素类型。
/// @tparam __Base 基础迭代器类型。
template<class T, class __Base> class FilterRange;

/// @brief 映射迭代器。
/// @tparam T 参数元素类型。
/// @tparam U 映射元素类型。
/// @tparam __Base 基础迭代器类型。
template<class T, class U, class __Base> class MapRange;

/// @brief 归约迭代器。
/// @tparam T 参数元素类型。
/// @tparam U 规约元素类型。
/// @tparam __Base 基础迭代器类型。
template<class T, class U, class __Base> class ReduceRange;

/// @brief 筛选谓词。
/// @tparam T 要筛选的元素类型。
template<class T> using Predicate = std::function<bool(T const &)>;

/// @brief 映射函数。
/// @tparam T 参数元素类型。
/// @tparam U 映射元素类型。
template<class T, class U> using MapFunction = std::function<U(T &&)>;

/// @brief 归约函数。
/// @tparam T 参数元素类型。
/// @tparam U 规约元素类型。
template<class T, class U> using ReduceFunction = std::function<U(U &&, T &&)>;

/// @brief 抽象迭代器类型。
/// @tparam T 迭代元素类型。
/// @tparam Self 实际迭代器类型。
template<class T, class Self>
class Range {
    /// @brief 生成一个实际迭代器类型的指针。
    /// @return 虽然使用 `dynamic_cast`，但这个转换是安全的。
    Self *self() { return dynamic_cast<Self *>(this); }

public:
    /// @brief 获取下一个元素。
    /// @return 如果迭代器还有元素，返回 `std::optional<T>`，否则返回 `std::nullopt`。
    virtual std::optional<T> next() = 0;

    /// @brief 以当前迭代器为基础，构造一个筛选迭代器。
    /// @param fn 筛选谓词。
    auto filter(Predicate<T> &&fn) {
        return FilterRange<T, Self>(
            std::move(*self()),
            std::forward<Predicate<T>>(fn));
    }

    /// @brief 以当前迭代器为基础，构造一个映射迭代器。
    /// @tparam U 输出迭代器的元素类型。
    /// @param fn 映射函数。
    template<class U>
    auto map(MapFunction<T, U> &&fn) {
        return MapRange<T, U, Self>(
            std::move(*self()),
            std::forward<MapFunction<T, U>>(fn));
    }

    /// @brief 以当前迭代器为基础，构造一个规约迭代器。
    /// @tparam U 输出迭代器的元素类型。
    /// @param fn 规约函数。
    template<class U>
    auto reduce(U &&init, ReduceFunction<T, U> &&fn) {
        return ReduceRange<T, U, Self>(
            std::move(*self()),
            std::forward<ReduceFunction<T, U>>(fn),
            std::forward<U>(init));
    }
};

/// @brief 迭代迭代器。
/// @tparam T 迭代元素类型。
/// @tparam Iter 符合 c++ 迭代器概念的类型。
template<class T, class Iter>
class IterRange final : public Range<T, IterRange<T, Iter>> {
    // c++ 迭代器的输出必须能转换到需要的元素类型。
    static_assert(std::is_convertible_v<typename std::iterator_traits<Iter>::value_type, T>);

    /// @brief 迭代器和尾后迭代器。
    Iter _it, _end;

public:
    /// @brief 构造迭代迭代器。
    /// @param it 迭代器。
    /// @param end 尾后迭代器。
    IterRange(Iter &&it, Iter &&end)
        : _it(std::forward<Iter>(it)),
          _end(std::forward<Iter>(end)) {}

    std::optional<T> next() override {
        if (_it != _end) {
            return {*_it++};
        } else {
            return {};
        }
    }
};

/// @brief 筛选迭代器。
/// @tparam T 筛选的元素类型。
/// @tparam __Base 基础迭代器类型。
template<class T, class __Base>
class FilterRange final : public Range<T, FilterRange<T, __Base>> {
    using __Predicate = Predicate<T>;

    __Predicate _fn;
    __Base _base;

public:
    FilterRange(__Base &&range, __Predicate &&fn)
        : _base(std::forward<__Base>(range)),
          _fn(std::forward<__Predicate>(fn)) {}

    std::optional<T> next() override {
        for (auto t = _base.next(); t; t = _base.next()) {
            if (_fn(*t)) {
                return {*t};
            }
        }
        return {};
    }
};

/// @brief 映射迭代器。
/// @tparam T 参数元素类型。
/// @tparam U 映射元素类型。
/// @tparam __Base 基础迭代器类型。
template<class T, class U, class __Base>
class MapRange final : public Range<U, MapRange<T, U, __Base>> {
    using __MapFunction = MapFunction<T, U>;

    __MapFunction _fn;
    __Base _base;

public:
    MapRange(__Base &&range, __MapFunction &&fn)
        : _base(std::forward<__Base>(range)),
          _fn(std::forward<__MapFunction>(fn)) {}

    std::optional<U> next() override {
        if (auto t = _base.next(); t) {
            return {_fn(std::move(*t))};
        } else {
            return {};
        }
    }
};

/// @brief 归约迭代器。
/// @tparam T 参数元素类型。
/// @tparam U 规约元素类型。
/// @tparam __Base 基础迭代器类型。
template<class T, class U, class __Base>
class ReduceRange final : public Range<U, ReduceRange<T, U, __Base>> {
    using __ReduceFunction = ReduceFunction<T, U>;

    __ReduceFunction _fn;
    __Base _base;
    U _val;

public:
    ReduceRange(__Base &&range, __ReduceFunction &&fn, U &&init)
        : _base(std::forward<__Base>(range)),
          _fn(std::forward<__ReduceFunction>(fn)),
          _val(std::forward<U>(init)) {}

    std::optional<U> next() override {
        if (auto t = _base.next()) {
            return _val = _fn(std::move(_val), std::move(*t));
        } else {
            return {};
        }
    }
};

/// @brief 生成一个迭代迭代器。
/// @tparam Iter 符合 c++ 迭代器概念的类型。
/// @param begin 迭代器。
/// @param end 尾后迭代器。
/// @return 迭代迭代器。
template<class Iter>
auto range(Iter begin, Iter end) {
    using value_type = typename std::iterator_traits<Iter>::value_type;
    return IterRange<value_type, Iter>(
        std::move(begin),
        std::move(end));
}
