#include <compress_x.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::buffer compress(py::array_t<double> original, double tol, double s,
                    mgard_x::error_bound_type error_bound_type, mgard_x::Config config)
{
    std::vector<mgard_x::SIZE> shape(original.shape(), original.shape() + original.ndim());
    void *compressed_data = nullptr;
    size_t compressed_size = 0;

    mgard_x::compress_status_type status = mgard_x::compress(
        original.ndim(), mgard_x::data_type::Double, shape, tol, s, error_bound_type,
        original.data(), compressed_data, compressed_size, config, false);

    switch (status) {
    case mgard_x::compress_status_type::Success:
        break;
    case mgard_x::compress_status_type::Failure:
        throw std::runtime_error("Compression failure");
        break;
    case mgard_x::compress_status_type::OutputTooLargeFailure:
        throw std::length_error("Output too large");
        break;
    case mgard_x::compress_status_type::NotSupportHigherNumberOfDimensionsFailure:
        throw std::invalid_argument("Not supported higher number of dimensions");
        break;
    case mgard_x::compress_status_type::NotSupportDataTypeFailure:
        throw std::invalid_argument("Not supported data type");
        break;
    case mgard_x::compress_status_type::BackendNotAvailableFailure:
        throw std::invalid_argument("Backend not available");
        break;
    }

    return py::array_t<unsigned char>({compressed_size}, {1},
                                      static_cast<unsigned char *>(compressed_data),
                                      py::capsule(compressed_data, [](void *ptr) { delete ptr; }));
}

py::array_t<double> decompress(py::buffer compressed, mgard_x::Config config)
{
    py::buffer_info info = compressed.request();

    if (info.format != py::format_descriptor<unsigned char>::format()) {
        throw std::invalid_argument("Input must be a byte array");
    }
    if (info.shape.size() != 1) {
        throw std::invalid_argument("Input must be a 1D array");
    }

    void *decompressed_data = nullptr;
    std::vector<mgard_x::SIZE> shape;
    mgard_x::data_type dtype;

    mgard_x::compress_status_type status =
        decompress(info.ptr, info.size, decompressed_data, shape, dtype, config, false);

    switch (status) {
    case mgard_x::compress_status_type::Success:
        break;
    case mgard_x::compress_status_type::Failure:
        throw std::runtime_error("Compression failure");
        break;
    case mgard_x::compress_status_type::NotSupportHigherNumberOfDimensionsFailure:
        throw std::invalid_argument("Not supported higher number of dimensions");
        break;
    case mgard_x::compress_status_type::NotSupportDataTypeFailure:
        throw std::invalid_argument("Not supported data type");
        break;
    case mgard_x::compress_status_type::BackendNotAvailableFailure:
        throw std::invalid_argument("Backend not available");
        break;
    }

    return py::array_t<double>(shape, reinterpret_cast<double *>(decompressed_data),
                               py::capsule(decompressed_data, [](void *ptr) { delete ptr; }));
}

PYBIND11_MODULE(_mgard, m)
{
    m.doc() = "MGARD Python bindings";

    py::enum_<mgard_x::device_type>(m, "DeviceType")
        .value("Auto", mgard_x::device_type::AUTO)
        .value("SERIAL", mgard_x::device_type::SERIAL)
        .value("OPENMP", mgard_x::device_type::OPENMP)
        .value("CUDA", mgard_x::device_type::CUDA)
        .value("HIP", mgard_x::device_type::HIP)
        .value("SYCL", mgard_x::device_type::SYCL);

    py::enum_<mgard_x::lossless_type>(m, "LosslessType")
        .value("Huffman", mgard_x::lossless_type::Huffman)
        .value("HuffmanLZ4", mgard_x::lossless_type::Huffman_LZ4)
        .value("HuffmanZstd", mgard_x::lossless_type::Huffman_Zstd);

    py::enum_<mgard_x::domain_decomposition_type>(m, "DomainDecompositionType")
        .value("MaxDim", mgard_x::domain_decomposition_type::MaxDim)
        .value("Block", mgard_x::domain_decomposition_type::Block);

    py::enum_<mgard_x::decomposition_type>(m, "DecompositionType")
        .value("MultiDim", mgard_x::decomposition_type::MultiDim)
        .value("SingleDim", mgard_x::decomposition_type::SingleDim);

    py::enum_<mgard_x::error_bound_type>(m, "ErrorBoundType")
        .value("REL", mgard_x::error_bound_type::REL)
        .value("ABS", mgard_x::error_bound_type::ABS);

    py::class_<mgard_x::Config>(m, "Config")
        .def(py::init())
        .def_readwrite("dev_type", &mgard_x::Config::dev_type)
        .def_readwrite("dev_id", &mgard_x::Config::dev_id)
        .def_readwrite("num_dev", &mgard_x::Config::num_dev)
        .def_readwrite("reorder", &mgard_x::Config::reorder)
        .def_readwrite("lossless", &mgard_x::Config::lossless)
        .def_readwrite("huff_dict_size", &mgard_x::Config::huff_dict_size)
        .def_readwrite("lz4_block_size", &mgard_x::Config::lz4_block_size)
        .def_readwrite("zstd_compress_level", &mgard_x::Config::zstd_compress_level)
        .def_readonly("normalize_coordinates", &mgard_x::Config::normalize_coordinates)
        .def_readwrite("domain_decomposition", &mgard_x::Config::domain_decomposition)
        .def_readwrite("decomposition", &mgard_x::Config::decomposition)
        .def_readwrite("max_larget_level", &mgard_x::Config::max_larget_level)
        .def_readwrite("prefetch", &mgard_x::Config::prefetch)
        .def_readwrite("max_memory_footprint", &mgard_x::Config::max_memory_footprint)
        .def_readwrite("adjust_shape", &mgard_x::Config::adjust_shape);

    m.def("compress", &compress, "Compress a multi-dimensional array", py::arg("original"),
          py::arg("tol"), py::arg("s"),
          py::arg("error_bound_type") = mgard_x::error_bound_type::REL,
          py::arg("config") = mgard_x::Config());
    m.def("decompress", &decompress, "Decompress a multi-dimensional array", py::arg("compressed"),
          py::arg("config") = mgard_x::Config());
}
