#pragma once
#include "tensor.h"
#include <cmath>

namespace rl_engine {

// y = (x - μ)/√(σ² + ε) * γ + β
// 沿最后一维做归一化
class LayerNorm {
    
private:
    size_t dim_;
    float  eps_;
    Tensor2D gamma_;
    Tensor2D beta_;
    
public:
    void init(const float* gamma, const float* beta, size_t dim, float eps = 1e-5f) {
        dim_  = dim;
        eps_  = eps;
        gamma_ = Tensor2D(1, dim);
        beta_  = Tensor2D(1, dim);
        gamma_.load_from(gamma, dim);
        beta_.load_from(beta, dim);
    }

    void init(size_t dim, float eps = 1e-5f) {
        dim_  = dim;
        eps_  = eps;
        gamma_ = Tensor2D(1, dim);
        beta_  = Tensor2D(1, dim);
    }

    void load(const float* gamma, const float* beta) {
        gamma_.load_from(gamma, dim_);
        beta_.load_from(beta, dim_);
    }

    Tensor2D forward(const Tensor2D& x) const {
        size_t batch = x.rows();
        size_t d     = x.cols();
        Tensor2D y(batch, d);

        for (size_t b = 0; b < batch; ++b) {
            // 计算均值
            float mean = 0.0f;
            for (size_t i = 0; i < d; ++i) mean += x(b, i);
            mean /= static_cast<float>(d);

            // 计算方差
            float var = 0.0f;
            for (size_t i = 0; i < d; ++i) {
                float diff = x(b, i) - mean;
                var += diff * diff;
            }
            var /= static_cast<float>(d);

            // 归一化 + 仿射
            float inv_std = 1.0f / std::sqrt(var + eps_);
            for (size_t i = 0; i < d; ++i) {
                y(b, i) = (x(b, i) - mean) * inv_std * gamma_(0, i) + beta_(0, i);
            }
        }
        return y;
    }

    size_t dim() const { return dim_; }

};

}  // namespace rl_engine
