#include <compress_x.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::buffer compress(py::array_t<double> original, double tol, double s)
{
    std::vector<mgard_x::SIZE> shape(original.shape(), original.shape() + original.ndim());
    void *compressed_data = nullptr;
    size_t compressed_size = 0;

    // TODO Accept mgard_x::Config
    mgard_x::compress_status_type status = mgard_x::compress(
        original.ndim(), mgard_x::data_type::Double, shape, tol, s, mgard_x::error_bound_type::REL,
        original.data(), compressed_data, compressed_size, false);

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
        throw std::invalid_argument("Backed not available");
        break;
    }

    return py::array_t<unsigned char>({compressed_size}, {1},
                                      static_cast<unsigned char *>(compressed_data),
                                      py::capsule(compressed_data, [](void *ptr) { delete ptr; }));
}

py::array_t<double> decompress(py::buffer compressed)
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

    // TODO Accept mgard_x::Config
    mgard_x::compress_status_type status =
        decompress(info.ptr, info.size, decompressed_data, shape, dtype, false);

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
        throw std::invalid_argument("Backed not available");
        break;
    }

    return py::array_t<double>(shape, reinterpret_cast<double *>(decompressed_data),
                               py::capsule(decompressed_data, [](void *ptr) { delete ptr; }));
}

PYBIND11_MODULE(pymgard, m)
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

    m.def("compress", &compress, "Compress a multi-dimensional array",
          py::return_value_policy::take_ownership);
    m.def("decompress", &decompress, "Decompress a multi-dimensional array",
          py::return_value_policy::take_ownership);
}
