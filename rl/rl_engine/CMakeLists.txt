cmake_minimum_required(VERSION 3.18)
project(rl_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    # /O2 和 /arch:AVX2 只在 Release 模式生效（Debug 的 /RTC1 不兼容优化选项）
    add_compile_options($<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/arch:AVX2>)
    # 强制 UTF-8 源码编码（避免中文注释在 GBK 下报 C4819）
    add_compile_options(/utf-8)
endif()

# 强制使用 Python 3.11（避免 CMake 抓到 WindowsApps 的 3.13）
set(PYTHON_EXECUTABLE "C:/Users/www19/AppData/Local/Programs/Python/Python311/python.exe" CACHE FILEPATH "Python 3.11 interpreter")
set(pybind11_DIR "C:/Users/www19/AppData/Local/Programs/Python/Python311/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)
include_directories(${CMAKE_SOURCE_DIR}/include)

pybind11_add_module(_rl_engine
    bindings/py_bindings.cpp
)

# 输出 _rl_engine.pyd 到 rl/（所有配置统一路径，不加 Debug/Release 子目录）
set_target_properties(_rl_engine PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/.."
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/.."
)