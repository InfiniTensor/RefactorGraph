#include "iterator/range.hpp"
#include <cmath>
#include <iostream>
#include <thread>

int main() {
    int arr[]{1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto ptr = range(arr, arr + 9)
                   .filter([](auto &&x) { return x % 2 == 0; })
                   .map<float>([](auto &&x) { return std::sqrt(x); })
                   .reduce<float>(0.0f, [](auto &&x, auto &&y) { return x + y; });
    while (auto x = ptr.next()) {
        std::cout << *x << std::endl;
    }
    return 0;
}
