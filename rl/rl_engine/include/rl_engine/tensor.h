#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace rl_engine {

// 简单的 2D 浮点张量，行主序（row-major）存储
// 设计理由：推理时 batch_size=1，不需要 3D/4D 张量
class Tensor2D {
    
private:
    size_t rows_, cols_;
    std::vector<float> data_;

public:
    Tensor2D() : rows_(0), cols_(0) {}
    Tensor2D(size_t rows, size_t cols) 
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0f) {}
    
    // ---- 访问 ----
    float& operator()(size_t r, size_t c) { return data_[r * cols_ + c]; }
    float  operator()(size_t r, size_t c) const { return data_[r * cols_ + c]; }
    float* data() { return data_.data(); }
    const float* data() const { return data_.data(); }
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t size() const { return data_.size(); }

    // ---- 创建 ----
    static Tensor2D zeros(size_t r, size_t c) { return Tensor2D(r, c); }
    static Tensor2D ones(size_t r, size_t c) {
        Tensor2D t(r, c);
        std::fill(t.data_.begin(), t.data_.end(), 1.0f);
        return t;
    }

    // ---- 从原始指针加载权重（pybind11 传过来的） ----
    void load_from(const float* src, size_t n) {
        if (n > data_.size()) throw std::runtime_error("Tensor2D::load_from: 数据过大");
        std::copy(src, src + n, data_.begin());
    }

    const std::vector<float>& vec() const { return data_; }

};

}  // namespace rl_engine
