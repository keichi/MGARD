#include <compress_x.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

py::buffer compress(py::array_t<double> original, double tol, double s)
{
    std::vector<mgard_x::SIZE> shape(original.shape(), original.shape() + original.ndim());
    void *compressed_data = nullptr;
    size_t compressed_size = 0;

    // TODO accept mgard_x::Config and check status
    mgard_x::compress_status_type status = mgard_x::compress(
        original.ndim(), mgard_x::data_type::Double, shape, tol, s, mgard_x::error_bound_type::REL,
        original.data(), compressed_data, compressed_size, false);

    // TODO is there a way to free compressed_data if the py::array is freed?
    py::array_t<unsigned char> compressed({static_cast<pybind11::ssize_t>(compressed_size)},
                                          reinterpret_cast<unsigned char *>(compressed_data));
    delete compressed_data;

    return compressed;
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

    // TODO accept mgard_x::Config and check status
    mgard_x::compress_status_type status =
        decompress(info.ptr, info.size, decompressed_data, shape, dtype, false);

    py::array_t<double> decompressed(shape, reinterpret_cast<double *>(decompressed_data));
    delete decompressed_data;

    return decompressed;
}

PYBIND11_MODULE(pymgard, m)
{
    m.doc() = "MGARD Python bindings";

    m.def("compress", &compress, "Compress a multi-dimensional array",
          py::return_value_policy::take_ownership);
    m.def("decompress", &decompress, "Decompress a multi-dimensional array",
          py::return_value_policy::take_ownership);
}
