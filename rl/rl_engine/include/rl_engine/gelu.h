#pragma once
#include "tensor.h"
#include <cmath>

namespace rl_engine {

// GELU 近似（tanh 版本，误差 < 0.1%）
// 精度公式: 0.5*x*(1 + tanh(√(2/π) * (x + 0.044715*x³)))
inline float gelu(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;  // √(2/π)
    constexpr float COEFF          = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + std::tanh(SQRT_2_OVER_PI * (x + COEFF * x3)));
}

// 逐元素 GELU
inline Tensor2D gelu_forward(const Tensor2D& x) {
    Tensor2D y(x.rows(), x.cols());
    for (size_t i = 0; i < x.size(); ++i) {
        y.data()[i] = gelu(x.data()[i]);
    }
    return y;
}

}  // namespace rl_engine
