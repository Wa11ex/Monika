#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "rl_engine/policy.h"

namespace py = pybind11;

PYBIND11_MODULE(_rl_engine, m) {
    m.doc() = "Monika RL inference engine (C++ backend)";

    py::class_<rl_engine::PolicyModel>(m, "PolicyModel")
        .def(py::init<>())
        .def("load", &rl_engine::PolicyModel::load)
        .def("forward", [](rl_engine::PolicyModel& self,
                           py::array_t<float, py::array::c_style> features) {
            auto buf = features.request();
            if (buf.size != 15)
                throw std::runtime_error("特征必须是 15 维");
            auto out = self.forward(static_cast<const float*>(buf.ptr));
            py::dict d;
            d["memory_count"] = out.memory_count;
            d["use_tool_prob"] = out.use_tool_prob;
            d["temperature"] = out.temperature;
            d["max_tokens"] = out.max_tokens;
            py::list tl, ml;
            for (int i = 0; i < 6; ++i) tl.append(out.tool_logits[i]);
            for (int i = 0; i < 3; ++i) ml.append(out.mode_logits[i]);
            d["tool_logits"] = tl;
            d["mode_logits"] = ml;
            return d;
        }, py::arg("features"));
}