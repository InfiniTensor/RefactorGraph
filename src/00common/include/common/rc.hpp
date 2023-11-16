#ifndef RC_HPP
#define RC_HPP

#include <cstddef>
#include <functional>
#include <utility>

namespace refactor {

    template<class T> class Rc;
    template<class T> class Weak;
    template<class T> class RefInplace;

    template<class T>
    class Rc {
        using element_type = std::remove_extent_t<T>;
        friend class Weak<T>;
        friend class RefInplace<T>;

        T *_value;
        struct Counter {
            size_t strong, weak;
        } * _counter;

        Rc(T *ptr, Counter *counter) noexcept
            : _value(ptr), _counter(counter) { inc(); }

        void inc() {
            if (_counter) { ++_counter->strong; }
        }
        void dec() {
            if (_counter && !--_counter->strong) {
                delete std::exchange(_value, nullptr);
                if (!_counter->weak) {
                    delete std::exchange(_counter, nullptr);
                }
            }
        }

    public:
        explicit Rc(T *ptr) noexcept
            : _value(ptr),
              _counter(nullptr) {
            if (!ptr) { return; }
            _counter = new Counter{1, 0};
            if (auto *in = dynamic_cast<RefInplace<T> *>(ptr); in) {
                ASSERT(!in->_counter, "");
                in->_counter = _counter;
            }
        }

        Rc(Weak<T> const &weak) noexcept
            : _value(weak._counter
                         ? (++weak._counter->strong, weak._value)
                         : nullptr),
              _counter(weak._counter) {}

        Rc() noexcept
            : _value(nullptr),
              _counter(nullptr) {}

        Rc(std::nullptr_t) noexcept : Rc() {}

        Rc(Rc const &rhs) noexcept
            : _value(rhs._value),
              _counter(rhs._counter) { inc(); }

        Rc(Rc &&rhs) noexcept
            : _value(std::exchange(rhs._value, nullptr)),
              _counter(std::exchange(rhs._counter, nullptr)) {}

        ~Rc() noexcept { dec(); }

        Rc &operator=(Rc const &rhs) noexcept {
            if (this != &rhs) {
                dec();
                _value = rhs._value;
                _counter = rhs._counter;
                inc();
            }
            return *this;
        }

        Rc &operator=(Rc &&rhs) noexcept {
            if (this != &rhs) {
                dec();
                _value = std::exchange(rhs._value, nullptr);
                _counter = std::exchange(rhs._counter, nullptr);
            }
            return *this;
        }

        operator bool() const noexcept { return _counter; }
        auto operator!() const noexcept -> bool { return !_counter; }
        auto operator==(Rc const &rhs) const noexcept -> bool { return _counter == rhs._counter; }
        auto operator!=(Rc const &rhs) const noexcept -> bool { return _counter != rhs._counter; }
        auto operator<(Rc const &rhs) const noexcept -> bool { return _counter < rhs._counter; }
        auto operator>(Rc const &rhs) const noexcept -> bool { return _counter > rhs._counter; }
        auto operator<=(Rc const &rhs) const noexcept -> bool { return _counter <= rhs._counter; }
        auto operator>=(Rc const &rhs) const noexcept -> bool { return _counter >= rhs._counter; }

        auto get() const noexcept -> element_type * { return _counter ? _value : nullptr; }
        auto operator*() const noexcept -> element_type & { return *_value; }
        auto operator->() const noexcept -> element_type * { return _value; }

        auto use_count() const noexcept -> size_t { return _counter ? _counter->strong : 0; }
    };

    template<class T>
    class Weak {
        using Counter = typename Rc<T>::Counter;
        friend class Rc<T>;

        T *_value;
        Counter *_counter;

        void inc() {
            if (_counter) { ++_counter->weak; }
        }
        void dec() {
            if (_counter && !--_counter->weak) {
                if (!_counter->strong) {
                    delete std::exchange(_counter, nullptr);
                }
            }
        }

    public:
        Weak(Rc<T> const &rc) noexcept
            : _value(rc._value),
              _counter(rc._counter) { inc(); }

        Weak() noexcept
            : _value(nullptr),
              _counter(nullptr) {}

        Weak(std::nullptr_t) noexcept : Weak() {}

        Weak(Weak const &rhs) noexcept
            : _value(rhs._value),
              _counter(rhs._counter) { inc(); }

        Weak(Weak &&rhs) noexcept
            : _value(rhs._value),
              _counter(std::exchange(rhs._counter, nullptr)) {}

        ~Weak() noexcept { dec(); }

        Weak &operator=(Weak const &rhs) noexcept {
            if (this != &rhs) {
                dec();
                _value = rhs._value;
                _counter = rhs._counter;
                inc();
            }
            return *this;
        }

        Weak &operator=(Weak &&rhs) noexcept {
            if (this != &rhs) {
                dec();
                _value = rhs._value;
                _counter = std::exchange(rhs._counter, nullptr);
            }
            return *this;
        }

        operator bool() const noexcept { return _counter; }
        auto operator!() const noexcept -> bool { return !_counter; }
        auto operator==(Weak const &rhs) const noexcept -> bool { return _counter == rhs._counter; }
        auto operator!=(Weak const &rhs) const noexcept -> bool { return _counter != rhs._counter; }
        auto operator<(Weak const &rhs) const noexcept -> bool { return _counter < rhs._counter; }
        auto operator>(Weak const &rhs) const noexcept -> bool { return _counter > rhs._counter; }
        auto operator<=(Weak const &rhs) const noexcept -> bool { return _counter <= rhs._counter; }
        auto operator>=(Weak const &rhs) const noexcept -> bool { return _counter >= rhs._counter; }

        auto use_count() const noexcept -> size_t { return _counter ? _counter->strong : 0; }
        auto expired() const noexcept -> bool { return _counter ? _counter->strong == 0 : true; }
        auto lock() const noexcept -> Rc<T> { return Rc<T>(*this); }
    };

    template<class T>
    class RefInplace {
        using Counter = typename Rc<T>::Counter;
        friend class Rc<T>;

        mutable Counter *_counter = nullptr;

    public:
        auto shared_from_this() noexcept -> Rc<T> { return Rc<T>(reinterpret_cast<T *>(this), _counter); }
        auto shared_from_this() const noexcept -> Rc<T const> { return Rc<T>(reinterpret_cast<T const *>(this), _counter); }
    };

}// namespace refactor

template<class T>
struct std::hash<refactor::Rc<T>> {
    std::size_t operator()(refactor::Rc<T> const &rc) const noexcept {
        return reinterpret_cast<std::size_t>(rc.get());
    }
};

#endif// RC_HPP
