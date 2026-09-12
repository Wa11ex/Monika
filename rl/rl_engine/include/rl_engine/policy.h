#pragma once
#include "tensor.h"
#include "linear.h"
#include "layernorm.h"
#include "gelu.h"
#include <string>
#include <fstream>

namespace rl_engine {

    struct PolicyOutput {
        float memory_count;      // [1, 5]
        float use_tool_prob;     // [0, 1]
        float temperature;       // [0.5, 1.5]
        float max_tokens;        // [128, 1024]
        float tool_logits[6];    // softmax 前的 logits
        float mode_logits[3];    // softmax 前的 logits
    };

    class PolicyModel {
        private:
            // 脊骨（2 层）
            Linear    fc1_, fc2_;
            LayerNorm ln1_, ln2_;
            // 6 个输出头
            Linear    head_memory_, head_use_tool_, head_temperature_;
            Linear    head_max_tokens_, head_tool_choice_, head_mode_;
        public:
            PolicyModel() = default;

            void init(){
                fc1_.init(15, 64);
                ln1_.init(64);
                fc2_.init(64, 32);
                ln2_.init(32);

                head_memory_.init(32, 1);
                head_use_tool_.init(32, 1);
                head_temperature_.init(32, 1);
                head_max_tokens_.init(32, 1);
                head_tool_choice_.init(32, 6);
                head_mode_.init(32, 3);
            };
            
            bool load(const std::string& path) {
                //这里正好规定一下文件：紧凑排列的float数组，顺序为：
                // fc1.weight, fc1.bias, ln1.gamma, ln1.beta,
                // fc2.weight, fc2.bias, ln2.gamma, ln2.beta,
                // head_memory.weight, head_memory.bias,
                // head_use_tool.weight, head_use_tool.bias,
                // head_temperature.weight, head_temperature.bias,
                // head_max_tokens.weight, head_max_tokens.bias,
                // head_tool_choice.weight, head_tool_choice.bias,
                // head_mode.weight, head_mode.bias
                
                std::ifstream file(path, std::ios::binary);
                if (!file.is_open()) {
                    return false;
                }

                auto read_tensor = [&file](std::vector<float>& t) {
                    float* data = t.data();
                    size_t bytes = t.size() * sizeof(float);
                    file.read(reinterpret_cast<char*>(data), bytes);
                };

                std::vector<float> buffer1;
                std::vector<float> buffer2;

                auto load_linear = [&read_tensor](std::vector<float>& b1, std::vector<float>& b2, Linear& linear){
                    b1.resize(linear.weight_size());
                    b2.resize(linear.bias_size());
                    read_tensor(b1);
                    read_tensor(b2);
                    linear.load(b1.data(), b2.data());
                };

                auto load_layernorm = [&read_tensor](std::vector<float>& b1, std::vector<float>& b2, LayerNorm& ln){
                    b1.resize(ln.dim());
                    b2.resize(ln.dim());
                    read_tensor(b1);
                    read_tensor(b2);
                    ln.load(b1.data(), b2.data());
                };

                load_linear(buffer1, buffer2, fc1_);
                load_layernorm(buffer1, buffer2, ln1_);
                load_linear(buffer1, buffer2, fc2_);
                load_layernorm(buffer1, buffer2, ln2_);
                load_linear(buffer1, buffer2, head_memory_);
                load_linear(buffer1, buffer2, head_use_tool_);
                load_linear(buffer1, buffer2, head_temperature_);
                load_linear(buffer1, buffer2, head_max_tokens_);
                load_linear(buffer1, buffer2, head_tool_choice_);
                load_linear(buffer1, buffer2, head_mode_);

                return file.good();
            };    // 从二进制文件加载权重
            PolicyOutput forward(const float* features_15d)
            const {

                Tensor2D x1(1, 15);
                Tensor2D y1(1, 64);
                Tensor2D y2(1, 32);
                PolicyOutput output;
                x1.load_from(features_15d, 15);

                x1=fc1_.forward(x1); //15->64
                y1=ln1_.forward(x1);
                y1=gelu_forward(y1);

                y2=fc2_.forward(y1); //64->32
                y2=ln2_.forward(y2);
                y2=gelu_forward(y2);

                output.memory_count = head_memory_.forward(y2)(0, 0);
                output.use_tool_prob = head_use_tool_.forward(y2)(0, 0);
                output.temperature = head_temperature_.forward(y2)(0, 0);
                output.max_tokens = head_max_tokens_.forward(y2)(0, 0);
                Tensor2D tool_logits = head_tool_choice_.forward(y2);
                for (int i = 0; i < 6; ++i) {
                    output.tool_logits[i] = tool_logits(0, i);
                }
                Tensor2D mode_logits = head_mode_.forward(y2);
                for (int i = 0; i < 3; ++i) {
                    output.mode_logits[i] = mode_logits(0, i);
                }

                return output;
            };  // 前向推理

    };

}