#pragma once
#include "tensor.h"

namespace rl_engine {

// y = xW + b
// x: (batch, input), W: (input, output), b: (1, output)
class Linear {

private:
    Tensor2D weight_;
    Tensor2D bias_;
    size_t in_features_;
    size_t out_features_;

public:
    Linear() = default;

    // 用已有的权重初始化（从 Python 传过来）
    void init(const Tensor2D& weight, const Tensor2D& bias) {
        weight_ = weight;
        bias_   = bias;
        in_features_  = weight_.rows();
        out_features_ = weight_.cols();
    }

    // 从两个 float 数组初始化
    void init(size_t in_features, size_t out_features,
              const float* w_data, const float* b_data) {
        in_features_ = in_features;
        out_features_ = out_features;
        weight_ = Tensor2D(in_features, out_features);
        weight_.load_from(w_data, in_features * out_features);
        bias_ = Tensor2D(1, out_features);
        bias_.load_from(b_data, out_features);
    }

    void init(size_t in_features, size_t out_features) {
        in_features_ = in_features;
        out_features_ = out_features;
        weight_ = Tensor2D(in_features, out_features);
        bias_ = Tensor2D(1, out_features);
    }

    void load(const float* w_data, const float* b_data) {
        weight_ = Tensor2D(in_features_, out_features_);
        weight_.load_from(w_data, in_features_ * out_features_);
        bias_ = Tensor2D(1, out_features_);
        bias_.load_from(b_data, out_features_);
    }

    Tensor2D forward(const Tensor2D& x) const {
        size_t batch = x.rows();
        size_t d_in  = x.cols();
        size_t d_out = weight_.cols();

        Tensor2D y(batch, d_out);
        // y = x @ W
        for (size_t b = 0; b < batch; ++b) {
            for (size_t j = 0; j < d_out; ++j) {
                float sum = 0.0f;
                for (size_t i = 0; i < d_in; ++i) {
                    sum += x(b, i) * weight_(i, j);
                }
                y(b, j) = sum + bias_(0, j);
            }
        }
        return y;
    }

    size_t in_features()  const { return in_features_; }
    size_t out_features() const { return out_features_; }
    size_t weight_size() const { return in_features_ * out_features_; }
    size_t bias_size()   const { return out_features_; }

};

}  // namespace rl_engine
