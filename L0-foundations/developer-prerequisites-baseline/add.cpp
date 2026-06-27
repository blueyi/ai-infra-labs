// add.cpp —— 用 pybind11 把 vector<float> 逐元素加法暴露给 Python
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // 关键：启用 std::vector <-> Python list 自动转换
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// 右值/移动友好：返回局部 vector 会被 NRVO 或 move，不产生大拷贝
std::vector<float> vector_add(const std::vector<float>& a,
                              const std::vector<float>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("两个向量长度必须一致");
    }
    std::vector<float> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = a[i] + b[i];   // SIMD 友好的连续访存：编译器可自动向量化
    }
    return out;                 // 移动语义：out 被搬走而非拷贝
}

// 模块定义：模块名 myadd 必须与 setup.py 中的扩展名一致
PYBIND11_MODULE(myadd, m) {
    m.doc() = "L0.5 demo: C++ vector add via pybind11";
    m.def("vector_add", &vector_add,
          "逐元素相加两个 float 向量",
          py::arg("a"), py::arg("b"));
}
