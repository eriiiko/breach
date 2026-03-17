#pragma once
// Grid2D<T> — contiguous row-major 2D array, zero-copy compatible with numpy.
// This is the shared memory format between Python and C++.

#include <vector>
#include <cstdint>
#include <cassert>

template<typename T>
class Grid2D {
    int h_, w_;
    std::vector<T> data_;

public:
    Grid2D() : h_(0), w_(0) {}
    Grid2D(int h, int w) : h_(h), w_(w), data_(h * w, T{}) {}

    // Wrap an external buffer (numpy's memory). No copy, no ownership.
    static Grid2D wrap(T* ptr, int h, int w) {
        Grid2D g;
        g.h_ = h;
        g.w_ = w;
        g.external_ = ptr;
        return g;
    }

    int height() const { return h_; }
    int width()  const { return w_; }
    int size()   const { return h_ * w_; }

    // Element access — (row, col) order, matching numpy [y, x]
    T& operator()(int y, int x)       { return raw()[y * w_ + x]; }
    T  operator()(int y, int x) const { return raw()[y * w_ + x]; }

    // Raw pointer for pybind11 buffer protocol and SIMD loops
    T*       raw()       { return external_ ? external_ : data_.data(); }
    const T* raw() const { return external_ ? external_ : data_.data(); }

    // Clamped access — returns the cell's own value if neighbor is out of bounds
    // (Neumann BC: zero gradient at boundary)
    T clamped(int y, int x) const {
        if (y < 0) y = 0;
        if (y >= h_) y = h_ - 1;
        if (x < 0) x = 0;
        if (x >= w_) x = w_ - 1;
        return raw()[y * w_ + x];
    }

private:
    T* external_ = nullptr;
};
