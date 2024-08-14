#include <iostream>
#include <vector>
#include <fstream>
#include <dnnl.hpp>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <cnpy.h>
using namespace dnnl;

// Template to read .npy files
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

// Template to write data to dnnl memory
inline void write_to_dnnl_memory(const void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        if (dst != nullptr) std::memcpy(dst, handle, size);
    }
}

// Template to read data from dnnl memory
inline void read_from_dnnl_memory(void *handle, const dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t size = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        if (src != nullptr) std::memcpy(handle, src, size);
    }
}

// Template to print tensor data
template <typename T>
void print_tensor(const std::vector<T> &tensor, const memory::dims &dims) {
    int b = dims[0];  // Batch size
    int c = dims[1];  // Number of channels
    int h = dims[2];  // Height
    int w = dims[3];  // Width

    std::cout << "Tensor data (" << b << "x" << c << "x" << h << "x" << w << "):\n";
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << "Channel " << j << ":\n";
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    std::cout << tensor[i * c * h * w + j * h * w + y * w + x] << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
}
memory conv_common_first_layer(engine &eng,
                stream &eng_stream,
                memory::dims &src_dims,
                memory::dims &weights_dims,
                memory::dims &bias_dims,
                memory::dims &dst_dims,
                int st, int pad,
                std::vector<std::string> npy_file
                )
{
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
    auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);


    // Create memory objects
    auto src_mem = memory(src_md, eng);
    auto weights_mem = memory(weights_md, eng);
    auto bias_mem = memory(bias_md, eng);
    auto conv_dst_mem = memory(dst_md, eng);

    auto weights_data = read_npy_file<float>(npy_file[1]);
    auto src_data = read_npy_file<float>(npy_file[0]);
    auto bias_data = read_npy_file<float>(npy_file[2]);

    // Write data to memory objects
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(weights_data.data(), weights_mem);
    write_to_dnnl_memory(bias_data.data(), bias_mem);

    // Create convolution primitive
    auto conv_pd = convolution_forward::primitive_desc(eng,
        prop_kind::forward_inference, algorithm::convolution_direct,
        src_md, weights_md, bias_md, dst_md,
        {st, st}, {pad, pad}, {pad, pad});

    auto conv_prim = convolution_forward(conv_pd);

    // Execute convolution primitive
    conv_prim.execute(eng_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_BIAS, bias_mem},
        {DNNL_ARG_DST, conv_dst_mem}
    });
    eng_stream.wait();
    return conv_dst_mem;
}

memory conv_common(engine &eng,
                stream &eng_stream,
                memory::dims &src_dims,
                memory::dims &weights_dims,
                memory::dims &bias_dims,
                memory::dims &dst_dims,
                memory &src_mem,
                int st, int pad,
                std::vector<std::string> npy_file
                )
{
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
    auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);


    // Create memory objects
    // auto src_mem = memory(src_md, eng);
    auto weights_mem = memory(weights_md, eng);
    auto bias_mem = memory(bias_md, eng);
    auto conv_dst_mem = memory(dst_md, eng);

    auto weights_data = read_npy_file<float>(npy_file[0]);
    auto bias_data = read_npy_file<float>(npy_file[1]);

    // Write data to memory objects

    write_to_dnnl_memory(weights_data.data(), weights_mem);
    write_to_dnnl_memory(bias_data.data(), bias_mem);

    // Create convolution primitive
    auto conv_pd = convolution_forward::primitive_desc(eng,
        prop_kind::forward_inference, algorithm::convolution_direct,
        src_md, weights_md, bias_md, dst_md,
        {st, st}, {pad, pad}, {pad, pad}); // Stride 2, Padding 3

    auto conv_prim = convolution_forward(conv_pd);

    // Execute convolution primitive
    conv_prim.execute(eng_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_BIAS, bias_mem},
        {DNNL_ARG_DST, conv_dst_mem}
    });
    eng_stream.wait();
    return conv_dst_mem;
}


memory maxpool_common(engine &eng,
                stream &eng_stream,
                memory::dims &pool_src_dims,
                memory::dims &pool_dst_dims,
                memory &pool_src_mem,
                int st, int ker, int dil, int pad_l, int pad_r
                )
{
    auto pool_src_md = memory::desc(pool_src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto pool_dst_md = memory::desc(pool_dst_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto pool_dst_mem = memory(pool_dst_md, eng);

    memory::dims strides = {st, st};
    memory::dims kernel = {ker, ker};
    memory::dims dilation = {dil, dil};
    memory::dims padding_l = {pad_l, pad_l};
    memory::dims padding_r = {pad_r, pad_r};

    auto pool_pd = pooling_forward::primitive_desc(
        eng,
        prop_kind::forward_inference,
        algorithm::pooling_max,
        pool_src_md,
        pool_dst_md,
        strides,
        kernel,
        dilation,
        padding_l,
        padding_r
    );

    auto pool_prim = pooling_forward(pool_pd);

    pool_prim.execute(eng_stream, {
        {DNNL_ARG_SRC, pool_src_mem},
        {DNNL_ARG_DST, pool_dst_mem}
    });
    eng_stream.wait();
    return pool_dst_mem;
}
memory add_common(engine &eng,
                stream &eng_stream,
                memory::dims &src_dims,
                memory &src_1_mem,
                memory &src_2_mem
                )
{
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    // auto pool_dst_md = memory::desc(pool_dst_dims, memory::data_type::f32, memory::format_tag::nchw);
    // auto src_1_mem = memory(src_md, eng);
    // auto src_2_mem = memory(src_md, eng);
    auto dst_mem = memory(src_md, eng);

    auto binary_pd = binary::primitive_desc(eng, algorithm::binary_add,
            src_md, src_md, src_md);

    // Create the primitive.
    auto binary_prim = binary(binary_pd);


    binary_prim.execute(eng_stream, {
        {DNNL_ARG_SRC_0, src_1_mem},
        {DNNL_ARG_SRC_1, src_2_mem},
        {DNNL_ARG_DST, dst_mem}
    });
    eng_stream.wait();
    return dst_mem;
}
memory relu_common(engine &eng,
                stream &eng_stream,
                memory &src_mem
                )
{
    auto relu_dst_mem = memory(src_mem.get_desc(), eng);
    auto relu_pd = eltwise_forward::primitive_desc(eng,
        prop_kind::forward_inference,
        algorithm::eltwise_relu,
        src_mem.get_desc(), src_mem.get_desc(),
        0.0f, 0.0f);  // alpha and beta for ReLU are typically 0

    auto relu_prim = eltwise_forward(relu_pd);

    relu_prim.execute(eng_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, relu_dst_mem}
    });
    eng_stream.wait();
    return relu_dst_mem;
}
memory reshape_common(engine &eng,
                stream &eng_stream,
                memory::dims &src_dims,
                memory::dims &reshape_dims,
                memory &src_mem
                )
{
    memory::desc src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    // auto src_mem = memory(src_md, eng);

    // Reshape the memory descriptor
    auto reshaped_md = src_md.reshape(reshape_dims);
    auto reshaped_mem = memory(reshaped_md, eng);
    reshaped_mem.set_data_handle(src_mem.get_data_handle()); // No data copy

    eng_stream.wait();
    return reshaped_mem;
}
memory reduce_mean_common(engine &eng,
                stream &eng_stream,
                memory::dims &src_dims,
                memory::dims &reshape_dims,
                memory::dims &reduce_dims,
                memory &src_mem
                )
{
    memory::desc src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    // auto src_mem = memory(src_md, eng);

    // Reshape the memory descriptor
    auto reshaped_md = src_md.reshape(reshape_dims);
    auto reshaped_mem = memory(reshaped_md, eng);
    reshaped_mem.set_data_handle(src_mem.get_data_handle()); // No data copy

    auto reduce_dims_md = memory::desc(reduce_dims, memory::data_type::f32, memory::format_tag::nc);
    // auto src_dims_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::any);
    auto reduce_mem = memory(reduce_dims_md, eng);

    auto reduction_pd = reduction::primitive_desc(eng,
                        algorithm::reduction_mean,
                        reshaped_md, reduce_dims_md,
                        0.0f, 0.0f
                        );
    auto reduction_prim = reduction(reduction_pd);

    reduction_prim.execute(eng_stream, {
        {DNNL_ARG_SRC, reshaped_mem},
        {DNNL_ARG_DST, reduce_mem}
    });
    eng_stream.wait();
    return reduce_mem;
}
memory global_avg_pool_common(engine &eng,
                stream &eng_stream,
                memory::dims &src_dims,
                memory::dims &dest_dims,
                memory &src_mem
                )
{
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto dst_md = memory::desc(dest_dims, memory::data_type::f32, memory::format_tag::nchw);
    std::cout <<"src_md " <<"dst_md"<<std::endl;
    auto dest_mem = memory(dst_md, eng);

    memory::dims kernel = {7, 7};  // Kernel size matches the spatial dimensions
    memory::dims strides = {1, 1}; // Stride doesn't matter for global pooling
    memory::dims padding = {0, 0};
    memory::dims dilation = {0, 0};

    auto pool_d = pooling_forward::primitive_desc(eng,
        prop_kind::forward_inference,
        algorithm::pooling_avg_exclude_padding,
        src_md, dst_md,
        strides, kernel,dilation , padding, padding);

    auto pool_prim = pooling_forward(pool_d);

    pool_prim.execute(eng_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_DST, dest_mem}
    });
    eng_stream.wait();
    return dest_mem;
}
memory gemm_common(engine &eng,
                stream &eng_stream,
                memory::dims &src_dims,
                memory::dims &weights_dims,
                memory::dims &bias_dims,
                memory::dims &dst_dims,
                memory &src_mem,
                std::vector<std::string> npy_file
                )
{
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nc);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oi);
    auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nc);

    // Create memory objects
    // auto src_mem = memory(src_md, eng);
    auto weights_mem = memory(weights_md, eng);
    auto bias_mem = memory(bias_md, eng);
    auto gemm_dst_mem = memory(dst_md, eng);

    auto weights_data = read_npy_file<float>(npy_file[0]);
    auto bias_data = read_npy_file<float>(npy_file[1]);

    // Write data to memory objects
    write_to_dnnl_memory(weights_data.data(), weights_mem);
    write_to_dnnl_memory(bias_data.data(), bias_mem);

    // Create convolution primitive
    auto gemm_pd = inner_product_forward::primitive_desc(eng,
        prop_kind::forward_inference,
        src_md, weights_md,bias_md, dst_md); // Stride 2, Padding 3

    auto gemm_prim = inner_product_forward(gemm_pd);

    // Execute convolution primitive
    gemm_prim.execute(eng_stream, {
        {DNNL_ARG_SRC, src_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_BIAS, bias_mem},
        {DNNL_ARG_DST, gemm_dst_mem}
    });
    eng_stream.wait();
    return gemm_dst_mem;
}
void save_to_npy(memory &dst_mem, memory::dims &dst_dims, const std::string &filename)
{
    std::vector<size_t> vec_dims(dst_dims.begin(), dst_dims.end());
    unsigned long total_size = 1;
    for (auto dim : dst_dims) {
        total_size*=dim;
    }
    std::vector<float> final_output_data(total_size);
    read_from_dnnl_memory(final_output_data.data(), dst_mem);

    std::string npy_filename = "./outputs/cpp_" + filename + ".npy";
    cnpy::npy_save(npy_filename, &final_output_data[0], vec_dims, "w");
}
void print_nodes(memory &dst_mem, std::string node_name,
    std::string op_type,
    memory::dims &ip_dims,
    memory::dims &op_dims)
{
    std::cout << "\n\n===== Node Name: " << node_name << " =====";
    std::cout << "\n Node OpType: " << op_type;
    std::cout << "\n Input Shape : ";
    for (auto dim : ip_dims) {
      std::cout << dim << " ";
    }
    std::cout << "\n Output Shape : ";
    for (auto dim : op_dims) {
      std::cout << dim << " ";
    }
    save_to_npy(dst_mem, op_dims, node_name);
}

void run_resnet18(){
       // Initialize engine and stream
        engine eng(engine::kind::cpu, 0);
        stream eng_stream(eng);
        //conv1 starts
        // Define tensor dimensions
        memory::dims src_dims = {1, 3, 224, 224};
        memory::dims weights_dims = {64, 3, 7, 7};
        memory::dims bias_dims = {64};
        memory::dims dst_dims = {1, 64, 112, 112};

        std::vector<std::string> input_npy_files = {"./inputs/py_input.npy",
                                                    "./weights/conv1_wt.npy",
                                                    "./weights/conv1_bs.npy"};
        auto conv_dst_mem = conv_common_first_layer(eng, eng_stream, 
                                                    src_dims, weights_dims,
                                                    bias_dims, dst_dims, 2, 3,
                                                    input_npy_files);
        print_nodes(conv_dst_mem, "conv1", "conv2d", src_dims, dst_dims);
        //conv1 ends
        //relu1 starts
        auto relu_1_dst_mem = relu_common(eng, eng_stream, conv_dst_mem);
        print_nodes(relu_1_dst_mem, "relu1", "relu", dst_dims, dst_dims);
        //relu1 ends
        //maxpool starts
        memory::dims pool_src_dims = {1, 64, 112, 112};
        memory::dims pool_dst_dims = {1, 64, 56, 56};
        auto pool_dst_mem = maxpool_common(eng, eng_stream,
                            pool_src_dims, pool_dst_dims,
                            relu_1_dst_mem, 2, 3, 0, 1, 1);
        print_nodes(pool_dst_mem, "maxpool", "maxpool", pool_src_dims, pool_dst_dims);
        //maxpool ends
        //conv2 starts
        memory::dims conv2_src_dims = {1, 64, 56, 56};
        memory::dims conv2_weights_dims = {64, 64, 3, 3};
        memory::dims conv2_bias_dims = {64};
        memory::dims conv2_dst_dims = {1, 64, 56, 56};
        std::vector<std::string> conv2_npy_files = {"./weights/conv2_wt.npy",
                                                    "./weights/conv2_bs.npy"};
        auto conv_dst_mem2 = conv_common(eng, eng_stream, 
                                            conv2_src_dims, conv2_weights_dims,
                                            conv2_bias_dims, conv2_dst_dims,
                                            pool_dst_mem,
                                            1, 1,
                                            conv2_npy_files);
        print_nodes(conv_dst_mem2, "conv2", "conv2d", conv2_src_dims, conv2_dst_dims);
        //conv2 ends
        //relu2 starts
        auto relu_2_dst_mem = relu_common(eng, eng_stream, conv_dst_mem2);
        print_nodes(relu_2_dst_mem, "relu2", "relu", conv2_dst_dims, conv2_dst_dims);
        //relu2 ends
        //conv3 starts
        memory::dims conv3_src_dims = {1, 64, 56, 56};
        memory::dims conv3_weights_dims = {64, 64, 3, 3};
        memory::dims conv3_bias_dims = {64};
        memory::dims conv3_dst_dims = {1, 64, 56, 56};
        std::vector<std::string> conv3_npy_files = {"./weights/conv3_wt.npy",
                                                    "./weights/conv3_bs.npy"};
        auto conv_dst_mem3 = conv_common(eng, eng_stream, 
                                            conv3_src_dims, conv3_weights_dims,
                                            conv3_bias_dims, conv3_dst_dims,
                                            relu_2_dst_mem,
                                            1, 1,
                                            conv3_npy_files);
        print_nodes(conv_dst_mem3, "conv3", "conv2d", conv3_src_dims, conv3_dst_dims);
        //conv3 ends
        //add1 starts
        auto add_dst_mem = add_common(eng, eng_stream,
                                        conv3_dst_dims,
                                        pool_dst_mem, conv_dst_mem3);
        print_nodes(add_dst_mem, "add1", "add", conv3_dst_dims, conv3_dst_dims);
        //add1 ends
        //relu3 starts
        auto relu_3_dst_mem = relu_common(eng, eng_stream, add_dst_mem);
        print_nodes(relu_3_dst_mem, "relu3", "relu", conv3_dst_dims, conv3_dst_dims);
        //relu3 ends
        //conv4 starts
        memory::dims conv4_src_dims = {1, 64, 56, 56};
        memory::dims conv4_weights_dims = {64, 64, 3, 3};
        memory::dims conv4_bias_dims = {64};
        memory::dims conv4_dst_dims = {1, 64, 56, 56};
        std::vector<std::string> conv4_npy_files = {"./weights/conv4_wt.npy",
                                                    "./weights/conv4_bs.npy"};
        auto conv_dst_mem4 = conv_common(eng, eng_stream,
                                            conv4_src_dims, conv4_weights_dims,
                                            conv4_bias_dims, conv4_dst_dims,
                                            relu_3_dst_mem,
                                            1, 1,
                                            conv4_npy_files);
        print_nodes(conv_dst_mem4, "conv4", "conv2d", conv4_src_dims, conv4_dst_dims);
        //conv4 ends
        //relu4 starts
        auto relu_4_dst_mem = relu_common(eng, eng_stream, conv_dst_mem4);
        print_nodes(relu_4_dst_mem, "relu4", "relu", conv4_dst_dims, conv4_dst_dims);
        //relu4 ends
        //conv5 starts
        memory::dims conv5_src_dims = {1, 64, 56, 56};
        memory::dims conv5_weights_dims = {64, 64, 3, 3};
        memory::dims conv5_bias_dims = {64};
        memory::dims conv5_dst_dims = {1, 64, 56, 56};
        std::vector<std::string> conv5_npy_files = {"./weights/conv5_wt.npy",
                                                    "./weights/conv5_bs.npy"};
        auto conv_dst_mem5 = conv_common(eng, eng_stream,
                                            conv5_src_dims, conv5_weights_dims,
                                            conv5_bias_dims, conv5_dst_dims,
                                            relu_4_dst_mem,
                                            1, 1,
                                            conv5_npy_files);
        print_nodes(conv_dst_mem5, "conv5", "conv2d", conv5_src_dims, conv5_dst_dims);
        //conv5 ends
        //add2 starts
        auto add_2_dst_mem = add_common(eng, eng_stream,
                                        conv5_dst_dims,
                                        relu_3_dst_mem, conv_dst_mem5);
        print_nodes(add_2_dst_mem, "add2", "add", conv5_dst_dims, conv5_dst_dims);
        //add2 ends
        //relu5 starts
        auto relu_5_dst_mem = relu_common(eng, eng_stream, add_2_dst_mem);
        print_nodes(relu_5_dst_mem, "relu5", "relu", conv5_dst_dims, conv5_dst_dims);
        //relu5 ends
        //conv6 starts
        memory::dims conv6_src_dims = {1, 64, 56, 56};
        memory::dims conv6_weights_dims = {128, 64, 3, 3};
        memory::dims conv6_bias_dims = {128};
        memory::dims conv6_dst_dims = {1, 128, 28, 28};
        std::vector<std::string> conv6_npy_files = {"./weights/conv6_wt.npy",
                                                    "./weights/conv6_bs.npy"};
        auto conv_dst_mem6 = conv_common(eng, eng_stream,
                                            conv6_src_dims, conv6_weights_dims,
                                            conv6_bias_dims, conv6_dst_dims,
                                            relu_5_dst_mem,
                                            2, 1,
                                            conv6_npy_files);
        print_nodes(conv_dst_mem6, "conv6", "conv2d", conv6_src_dims, conv6_dst_dims);
        //conv6 ends
        //relu6 starts
        auto relu_6_dst_mem = relu_common(eng, eng_stream, conv_dst_mem6);
        print_nodes(relu_6_dst_mem, "relu6", "relu", conv6_dst_dims, conv6_dst_dims);
        //relu6 ends
        //conv7 starts
        memory::dims conv7_src_dims = {1, 128, 28, 28};
        memory::dims conv7_weights_dims = {128, 128, 3, 3};
        memory::dims conv7_bias_dims = {128};
        memory::dims conv7_dst_dims = {1, 128, 28, 28};
        std::vector<std::string> conv7_npy_files = {"./weights/conv7_wt.npy",
                                                    "./weights/conv7_bs.npy"};
        auto conv_dst_mem7 = conv_common(eng, eng_stream,
                                            conv7_src_dims, conv7_weights_dims,
                                            conv7_bias_dims, conv7_dst_dims,
                                            relu_6_dst_mem,
                                            1, 1,
                                            conv7_npy_files);
        print_nodes(conv_dst_mem7, "conv7", "conv2d", conv7_src_dims, conv7_dst_dims);
        //conv7 ends
        //conv8 starts
        memory::dims conv8_src_dims = {1, 64, 56, 56};
        memory::dims conv8_weights_dims = {128, 64, 1, 1};
        memory::dims conv8_bias_dims = {128};
        memory::dims conv8_dst_dims = {1, 128, 28, 28};
        std::vector<std::string> conv8_npy_files = {"./weights/conv8_wt.npy",
                                                    "./weights/conv8_bs.npy"};
        auto conv_dst_mem8 = conv_common(eng, eng_stream,
                                            conv8_src_dims, conv8_weights_dims,
                                            conv8_bias_dims, conv8_dst_dims,
                                            relu_5_dst_mem,
                                            2, 0,
                                            conv8_npy_files);
        print_nodes(conv_dst_mem8, "conv8", "conv2d", conv8_src_dims, conv8_dst_dims);
        //conv8 ends
        //add3 starts
        auto add_3_dst_mem = add_common(eng, eng_stream,
                                        conv8_dst_dims, conv_dst_mem7, conv_dst_mem8);
        print_nodes(add_3_dst_mem, "add3", "add", conv8_dst_dims, conv8_dst_dims);
        //add3 ends
        //relu7 starts
        auto relu_7_dst_mem = relu_common(eng, eng_stream, add_3_dst_mem);
        print_nodes(relu_7_dst_mem, "relu7", "relu", conv8_dst_dims, conv8_dst_dims);
        //relu7 ends
        //conv9 starts
        memory::dims conv9_src_dims = {1, 128, 28, 28};
        memory::dims conv9_weights_dims = {128, 128, 3, 3};
        memory::dims conv9_bias_dims = {128};
        memory::dims conv9_dst_dims = {1, 128, 28, 28};
        std::vector<std::string> conv9_npy_files = {"./weights/conv9_wt.npy",
                                                    "./weights/conv9_bs.npy"};
        auto conv_dst_mem9 = conv_common(eng, eng_stream,
                                            conv9_src_dims, conv9_weights_dims,
                                            conv9_bias_dims, conv9_dst_dims,
                                            relu_7_dst_mem,
                                            1, 1,
                                            conv9_npy_files);
        print_nodes(conv_dst_mem9, "conv9", "conv2d", conv9_src_dims, conv9_dst_dims);
        //conv9 ends
        //relu8 starts
        auto relu_8_dst_mem = relu_common(eng, eng_stream, conv_dst_mem9);
        print_nodes(relu_8_dst_mem, "relu8", "relu", conv9_dst_dims, conv9_dst_dims);
        //relu8 ends
        //conv10 starts
        memory::dims conv10_src_dims = {1, 128, 28, 28};
        memory::dims conv10_weights_dims = {128, 128, 3, 3};
        memory::dims conv10_bias_dims = {128};
        memory::dims conv10_dst_dims = {1, 128, 28, 28};
        std::vector<std::string> conv10_npy_files = {"./weights/conv10_wt.npy",
                                                    "./weights/conv10_bs.npy"};
        auto conv_dst_mem10 = conv_common(eng, eng_stream,
                                            conv10_src_dims, conv10_weights_dims,
                                            conv10_bias_dims, conv10_dst_dims,
                                            relu_8_dst_mem,
                                            1, 1,
                                            conv10_npy_files);
        print_nodes(conv_dst_mem10, "conv10", "conv2d", conv10_src_dims, conv10_dst_dims);
        //conv10 ends
        //add4 starts
        auto add_4_dst_mem = add_common(eng, eng_stream,
                                        conv10_dst_dims, relu_7_dst_mem, conv_dst_mem10);
        print_nodes(add_4_dst_mem, "add4", "add", conv10_dst_dims, conv10_dst_dims);
        //add4 ends
        //relu9 starts
        auto relu_9_dst_mem = relu_common(eng, eng_stream, add_4_dst_mem);
        print_nodes(relu_9_dst_mem, "relu9", "relu", conv10_dst_dims, conv10_dst_dims);
        //relu9 ends
        //conv11 starts
        memory::dims conv11_src_dims = {1, 128, 28, 28};
        memory::dims conv11_weights_dims = {256, 128, 3, 3};
        memory::dims conv11_bias_dims = {256};
        memory::dims conv11_dst_dims = {1, 256, 14, 14};
        std::vector<std::string> conv11_npy_files = {"./weights/conv11_wt.npy",
                                                    "./weights/conv11_bs.npy"};
        auto conv_dst_mem11 = conv_common(eng, eng_stream,
                                            conv11_src_dims, conv11_weights_dims,
                                            conv11_bias_dims, conv11_dst_dims,
                                            relu_9_dst_mem,
                                            2, 1,
                                            conv11_npy_files);
        print_nodes(conv_dst_mem11, "conv11", "conv2d", conv11_src_dims, conv11_dst_dims);
        //conv11 ends
        //relu10 starts
        auto relu_10_dst_mem = relu_common(eng, eng_stream, conv_dst_mem11);
        print_nodes(relu_10_dst_mem, "relu10", "relu", conv11_dst_dims, conv11_dst_dims);
        //relu10 ends
        //conv12 starts
        memory::dims conv12_src_dims = {1, 256, 14, 14};
        memory::dims conv12_weights_dims = {256, 256, 3, 3};
        memory::dims conv12_bias_dims = {256};
        memory::dims conv12_dst_dims = {1, 256, 14, 14};
        std::vector<std::string> conv12_npy_files = {"./weights/conv12_wt.npy",
                                                    "./weights/conv12_bs.npy"};
        auto conv_dst_mem12 = conv_common(eng, eng_stream,
                                            conv12_src_dims, conv12_weights_dims,
                                            conv12_bias_dims, conv12_dst_dims,
                                            relu_10_dst_mem,
                                            1, 1,
                                            conv12_npy_files);
        print_nodes(conv_dst_mem12, "conv12", "conv2d", conv12_src_dims, conv12_dst_dims);
        //conv12 ends
        //conv13 starts
        memory::dims conv13_src_dims = {1, 128, 28, 28};
        memory::dims conv13_weights_dims = {256, 128, 1, 1};
        memory::dims conv13_bias_dims = {256};
        memory::dims conv13_dst_dims = {1, 256, 14, 14};
        std::vector<std::string> conv13_npy_files = {"./weights/conv13_wt.npy",
                                                    "./weights/conv13_bs.npy"};
        auto conv_dst_mem13 = conv_common(eng, eng_stream,
                                            conv13_src_dims, conv13_weights_dims,
                                            conv13_bias_dims, conv13_dst_dims,
                                            relu_9_dst_mem,
                                            2, 0,
                                            conv13_npy_files);
        print_nodes(conv_dst_mem13, "conv13", "conv2d", conv13_src_dims, conv13_dst_dims);
        //conv13 ends
        //add5 starts
        auto add_5_dst_mem = add_common(eng, eng_stream,
                                        conv12_dst_dims, conv_dst_mem12, conv_dst_mem13);
        print_nodes(add_5_dst_mem, "add5", "add", conv12_dst_dims, conv12_dst_dims);
        //add5 ends
        //relu11 starts
        auto relu_11_dst_mem = relu_common(eng, eng_stream, add_5_dst_mem);
        print_nodes(relu_11_dst_mem, "relu11", "relu", conv13_dst_dims, conv13_dst_dims);
        //relu11 ends
        //conv14 starts
        memory::dims conv14_src_dims = {1, 256, 14, 14};
        memory::dims conv14_weights_dims = {256, 256, 3, 3};
        memory::dims conv14_bias_dims = {256};
        memory::dims conv14_dst_dims = {1, 256, 14, 14};
        std::vector<std::string> conv14_npy_files = {"./weights/conv14_wt.npy",
                                                    "./weights/conv14_bs.npy"};
        auto conv_dst_mem14 = conv_common(eng, eng_stream,
                                            conv14_src_dims, conv14_weights_dims,
                                            conv14_bias_dims, conv14_dst_dims,
                                            relu_11_dst_mem,
                                            1, 1,
                                            conv14_npy_files);
        print_nodes(conv_dst_mem14, "conv14", "conv2d", conv14_src_dims, conv14_dst_dims);
        //conv14 ends
        //relu12 starts
        auto relu_12_dst_mem = relu_common(eng, eng_stream, conv_dst_mem14);
        print_nodes(relu_12_dst_mem, "relu12", "relu", conv14_dst_dims, conv14_dst_dims);
        //relu12 ends
        //conv15 starts
        memory::dims conv15_src_dims = {1, 256, 14, 14};
        memory::dims conv15_weights_dims = {256, 256, 3, 3};
        memory::dims conv15_bias_dims = {256};
        memory::dims conv15_dst_dims = {1, 256, 14, 14};
        std::vector<std::string> conv15_npy_files = {"./weights/conv15_wt.npy",
                                                    "./weights/conv15_bs.npy"};
        auto conv_dst_mem15 = conv_common(eng, eng_stream,
                                            conv15_src_dims, conv15_weights_dims,
                                            conv15_bias_dims, conv15_dst_dims,
                                            relu_12_dst_mem,
                                            1, 1,
                                            conv15_npy_files);
        print_nodes(conv_dst_mem15, "conv15", "conv2d", conv15_src_dims, conv15_dst_dims);
        //conv15 ends
        //add6 starts
        auto add_6_dst_mem = add_common(eng, eng_stream,
                                        conv15_dst_dims, relu_11_dst_mem, conv_dst_mem15);
        print_nodes(add_6_dst_mem, "add6", "add", conv15_dst_dims, conv15_dst_dims);
        //add6 ends
        //relu13 starts
        auto relu_13_dst_mem = relu_common(eng, eng_stream, add_6_dst_mem);
        print_nodes(relu_13_dst_mem, "relu13", "relu", conv15_dst_dims, conv15_dst_dims);
        //relu13 ends
        //conv16 starts
        memory::dims conv16_src_dims = {1, 256, 14, 14};
        memory::dims conv16_weights_dims = {512, 256, 3, 3};
        memory::dims conv16_bias_dims = {512};
        memory::dims conv16_dst_dims = {1, 512, 7, 7};
        std::vector<std::string> conv16_npy_files = {"./weights/conv16_wt.npy",
                                                    "./weights/conv16_bs.npy"};
        auto conv_dst_mem16 = conv_common(eng, eng_stream,
                                            conv16_src_dims, conv16_weights_dims,
                                            conv16_bias_dims, conv16_dst_dims,
                                            relu_13_dst_mem,
                                            2, 1,
                                            conv16_npy_files);
        print_nodes(conv_dst_mem16, "conv16", "conv2d", conv16_src_dims, conv16_dst_dims);
        //conv16 ends
        //relu14 starts
        auto relu_14_dst_mem = relu_common(eng, eng_stream, conv_dst_mem16);
        print_nodes(relu_14_dst_mem, "relu14", "relu", conv16_dst_dims, conv16_dst_dims);
        //relu14 ends
        //conv17 starts
        memory::dims conv17_src_dims = {1, 512, 7, 7};
        memory::dims conv17_weights_dims = {512, 512, 3, 3};
        memory::dims conv17_bias_dims = {512};
        memory::dims conv17_dst_dims = {1, 512, 7, 7};
        std::vector<std::string> conv17_npy_files = {"./weights/conv17_wt.npy",
                                                    "./weights/conv17_bs.npy"};
        auto conv_dst_mem17 = conv_common(eng, eng_stream,
                                            conv17_src_dims, conv17_weights_dims,
                                            conv17_bias_dims, conv17_dst_dims,
                                            relu_14_dst_mem,
                                            1, 1,
                                            conv17_npy_files);
        print_nodes(conv_dst_mem17, "conv17", "conv2d", conv17_src_dims, conv17_dst_dims);
        //conv17 ends
        //conv18 starts
        memory::dims conv18_src_dims = {1, 256, 14, 14};
        memory::dims conv18_weights_dims = {512, 256, 1, 1};
        memory::dims conv18_bias_dims = {512};
        memory::dims conv18_dst_dims = {1, 512, 7, 7};
        std::vector<std::string> conv18_npy_files = {"./weights/conv18_wt.npy",
                                                    "./weights/conv18_bs.npy"};
        auto conv_dst_mem18 = conv_common(eng, eng_stream,
                                            conv18_src_dims, conv18_weights_dims,
                                            conv18_bias_dims, conv18_dst_dims,
                                            relu_13_dst_mem,
                                            2, 0,
                                            conv18_npy_files);
        print_nodes(conv_dst_mem18, "conv18", "conv2d", conv18_src_dims, conv18_dst_dims);
        //conv18 ends
        //add7 starts
        auto add_7_dst_mem = add_common(eng, eng_stream,
                                        conv17_dst_dims, conv_dst_mem17, conv_dst_mem18);
        print_nodes(add_7_dst_mem, "add7", "add", conv18_dst_dims, conv18_dst_dims);
        //add7 ends
        //relu15 starts
        auto relu_15_dst_mem = relu_common(eng, eng_stream, add_7_dst_mem);
        print_nodes(relu_15_dst_mem, "relu15", "relu", conv18_dst_dims, conv18_dst_dims);
        //relu15 ends
        //conv19 starts
        memory::dims conv19_src_dims = {1, 512, 7, 7};
        memory::dims conv19_weights_dims = {512, 512, 3, 3};
        memory::dims conv19_bias_dims = {512};
        memory::dims conv19_dst_dims = {1, 512, 7, 7};
        std::vector<std::string> conv19_npy_files = {"./weights/conv19_wt.npy",
                                                    "./weights/conv19_bs.npy"};
        auto conv_dst_mem19 = conv_common(eng, eng_stream,
                                            conv19_src_dims, conv19_weights_dims,
                                            conv19_bias_dims, conv19_dst_dims,
                                            relu_15_dst_mem,
                                            1, 1,
                                            conv19_npy_files);
        print_nodes(conv_dst_mem19, "conv19", "conv2d", conv19_src_dims, conv19_dst_dims);
        //conv19 ends
        //relu16 starts
        auto relu_16_dst_mem = relu_common(eng, eng_stream, conv_dst_mem19);
        print_nodes(relu_16_dst_mem, "relu16", "relu", conv19_dst_dims, conv19_dst_dims);
        //relu16 ends
        //conv20 starts
        memory::dims conv20_src_dims = {1, 512, 7, 7};
        memory::dims conv20_weights_dims = {512, 512, 3, 3};
        memory::dims conv20_bias_dims = {512};
        memory::dims conv20_dst_dims = {1, 512, 7, 7};
        std::vector<std::string> conv20_npy_files = {"./weights/conv20_wt.npy",
                                                    "./weights/conv20_bs.npy"};
        auto conv_dst_mem20 = conv_common(eng, eng_stream,
                                            conv20_src_dims, conv20_weights_dims,
                                            conv20_bias_dims, conv20_dst_dims,
                                            relu_16_dst_mem,
                                            1, 1,
                                            conv20_npy_files);
        print_nodes(conv_dst_mem20, "conv20", "conv2d", conv20_src_dims, conv20_dst_dims);
        //conv20 ends
        //add8 starts
        auto add_8_dst_mem = add_common(eng, eng_stream,
                                        conv20_dst_dims, relu_15_dst_mem, conv_dst_mem20);
        print_nodes(add_8_dst_mem, "add8", "add", conv20_dst_dims, conv20_dst_dims);
        //add8 ends
        //relu17 starts
        auto relu_17_dst_mem = relu_common(eng, eng_stream, add_8_dst_mem);
        print_nodes(relu_17_dst_mem, "relu17", "relu", conv20_dst_dims, conv20_dst_dims);
        //relu17 ends
        // global avg pool starts
        memory::dims gpool_src_dims = {1, 512, 7, 7};
        memory::dims gpool_dst_dims = {1, 512, 1, 1};
        auto gpool_dst_mem = global_avg_pool_common(eng, eng_stream, gpool_src_dims,
                                    gpool_dst_dims,
                                    relu_17_dst_mem);
        print_nodes(gpool_dst_mem, "gpool", "global_avg_pool", gpool_src_dims, gpool_dst_dims);
        //global avg pool ends
        // reshape starts
        memory::dims reshape_dims = {1, 512};
        memory::desc reshape_src_md = memory::desc(gpool_dst_dims, memory::data_type::f32, memory::format_tag::nchw);

        // Reshape the memory descriptor
        auto reshaped_md = reshape_src_md.reshape(reshape_dims);
        auto reshaped_mem = memory(reshaped_md, eng);
        reshaped_mem.set_data_handle(gpool_dst_mem.get_data_handle()); // No data copy
        // reshape ends
        //Gemm starts
        memory::dims gemm_src_dims = {1, 512};
        memory::dims gemm_weights_dims = {10, 512};
        memory::dims gemm_bias_dims = {10};
        memory::dims gemm_dst_dims = {1, 10};
        std::vector<std::string> gemm_npy_files = {"./weights/gemm1_wt.npy",
                                                    "./weights/gemm1_bs.npy"};
        auto gemm_dst_mem = gemm_common(eng, eng_stream,
                                        gemm_src_dims, gemm_weights_dims,
                                        gemm_bias_dims, gemm_dst_dims,
                                        reshaped_mem,
                                        gemm_npy_files);
        print_nodes(gemm_dst_mem, "gemm", "gemm", gemm_src_dims, gemm_dst_dims);
        //Gemm ends

        // Update the final output to be the output of the fully connected layer
        std::vector<float> final_output_data(gemm_dst_dims[0] * gemm_dst_dims[1]);
        read_from_dnnl_memory(final_output_data.data(), gemm_dst_mem);
        std::cout << "\n\n==== Output ====" << std::endl;
        std::cout << "First 10 elements of final output: " << std::endl;
        for (int i = 0; i < 10 && i < final_output_data.size(); ++i) {
            std::cout << "   [" << final_output_data[i] << "]\n";
        }
        std::cout << std::endl;
}
int main() {
    try {
        run_resnet18();
    } catch (const dnnl::error &e) {
        std::cerr << "oneDNN error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

