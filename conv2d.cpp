#include <iostream>
#include <vector>
#include <fstream>
#include <dnnl.hpp>
#include <stdexcept>
#include <cassert>
#include <cstring>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <iomanip>

using namespace dnnl;
//  Read .npy files 
template <typename T>
std::vector<T> read_npy_file(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Read the entire file into a buffer
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(fileSize);
    file.read(buffer.data(), fileSize);

    // Check the .npy format
    if (fileSize < 10 || std::string(buffer.data(), 6) != "\x93NUMPY") {
        throw std::runtime_error("Invalid .npy file");
    }

    // Read version
    uint8_t major_version = buffer[6];
    uint8_t minor_version = buffer[7];

    // Read header length
    uint16_t header_len;
    if (major_version == 1) {
        header_len = *reinterpret_cast<uint16_t*>(&buffer[8]);
    } else if (major_version == 2) {
        header_len = *reinterpret_cast<uint32_t*>(&buffer[8]);
    } else {
        throw std::runtime_error("Unsupported .npy version");
    }

    // Read the header
    std::string header_str(&buffer[10], header_len);

    // Parse the header to get the shape and dtype
    size_t pos = header_str.find("'descr'") + 9;
    std::string dtype = header_str.substr(pos, header_str.find("'", pos) - pos);

    pos = header_str.find("'shape'") + 8;
    std::string shape_str = header_str.substr(pos, header_str.find(")", pos) - pos + 1);

    // Parse shape string
    std::vector<size_t> shape;
    size_t start = shape_str.find("(") + 1;
    size_t end = shape_str.find(")");
    std::string dims = shape_str.substr(start, end - start);
    size_t dim;
    std::istringstream shape_stream(dims);
    while (shape_stream >> dim) {
        shape.push_back(dim);
        if (shape_stream.peek() == ',')
            shape_stream.ignore();
    }

    // Calculate total size
    size_t total_size = 1;
    for (size_t s : shape) {
        total_size *= s;
    }

    // Read the data
    size_t data_start = 10 + header_len;
    size_t data_size = fileSize - data_start;

    if (data_size != total_size * sizeof(T)) {
        throw std::runtime_error("Data size mismatch");
    }

    std::vector<T> data(total_size);
    std::memcpy(data.data(), &buffer[data_start], data_size);

    // Check endianness and swap bytes if necessary
    if (dtype.find(">") != std::string::npos) {
        for (size_t i = 0; i < total_size; ++i) {
            char* p = reinterpret_cast<char*>(&data[i]);
            std::reverse(p, p + sizeof(T));
        }
    }

    return data;
}

// Read Binary files 
template <typename T>
std::vector<T> read_binary_file(const std::string &filename, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    std::vector<T> data(size);
    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    
    // Convert uint8_t to float and normalize to [0, 1] range
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(buffer[i]) / 255.0f;
    }
    return data;
}

// Write to DNNL Memory
inline void write_to_dnnl_memory(const void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        if (dst != nullptr) std::memcpy(dst, handle, size);
    }
}

// Read from DNNL Memory
inline void read_from_dnnl_memory(void *handle, const dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        if (src != nullptr) std::memcpy(handle, src, size);
    }
}

int main() {
    try {
        // Initialize engine and stream
        engine eng(engine::kind::cpu, 0);
        stream eng_stream(eng);

        // Define tensor dimensions
        memory::dims src_dims = {1, 1, 24, 24};
        memory::dims weights_dims = {64, 1, 3, 3};
        memory::dims bias_dims = {64};
        memory::dims dst_dims = {1, 64, 24, 24};
        
        // Load weights, biases, and batch normalization parameters from .npy files
        auto weights_data = read_npy_file<float>("python_weights.npy");
        auto bias_data = read_npy_file<float>("python_biases.npy");
        
        // Load MNIST image
        auto src_data = read_binary_file<float>("mnist_image.bin", src_dims[0] * src_dims[1] * src_dims[2] * src_dims[3]);

        // Create memory descriptors
        auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
        auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
        auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
        auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

        // Create memory objects
        auto src_mem = memory(src_md, eng);
        auto weights_mem = memory(weights_md, eng);
        auto bias_mem = memory(bias_md, eng);
        auto conv_dst_mem = memory(dst_md, eng);

        // Write data to memory objects
        write_to_dnnl_memory(src_data.data(), src_mem);
        write_to_dnnl_memory(weights_data.data(), weights_mem);
        write_to_dnnl_memory(bias_data.data(), bias_mem);

        // Create convolution primitive
        auto conv_pd = convolution_forward::primitive_desc(eng,
            prop_kind::forward_inference, algorithm::convolution_direct,
            src_md, weights_md, bias_md, dst_md,
            {1, 1}, {1, 1}, {1, 1});
        auto conv_prim = convolution_forward(conv_pd);

      
        // Execute convolution primitive
        conv_prim.execute(eng_stream, {{DNNL_ARG_SRC, src_mem},
                                       {DNNL_ARG_WEIGHTS, weights_mem},
                                       {DNNL_ARG_BIAS, bias_mem},
                                       {DNNL_ARG_DST, conv_dst_mem}});

        eng_stream.wait();

        // Read the output data after convolution
        std::vector<float> conv_output_data(dst_dims[0] * dst_dims[1] * dst_dims[2] * dst_dims[3]);
        read_from_dnnl_memory(conv_output_data.data(), conv_dst_mem);

        // Save the convolution output to a file
        std::ofstream conv_outfile("cpp_output_conv.bin", std::ios::binary);
        conv_outfile.write(reinterpret_cast<const char*>(conv_output_data.data()), conv_output_data.size() * sizeof(float));
        conv_outfile.close();

        std::cout << "C++ output after convolution saved to cpp_output_conv.bin" << std::endl;

    } catch (const dnnl::error &e) {
        std::cerr << "oneDNN error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}