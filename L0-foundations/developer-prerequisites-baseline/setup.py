# setup.py
from setuptools import setup, Extension
import pybind11

ext = Extension(
    name="myadd",                       # 必须与 PYBIND11_MODULE(myadd, ...) 一致
    sources=["add.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["-O3", "-std=c++17"],
)

setup(
    name="myadd",
    version="0.1.0",
    ext_modules=[ext],
)
