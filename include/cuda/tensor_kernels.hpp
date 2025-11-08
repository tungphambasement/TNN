#pragma once

#include <cstddef>

namespace cuda {

// Im2col/Col2im operations
template <typename T>
void im2col(const T *input, T *col_data, size_t batch_size, size_t channels, size_t height,
            size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w, size_t output_h, size_t output_w);

template <typename T>
void col2im(const T *col_data, T *output, size_t batch_size, size_t channels, size_t height,
            size_t width, size_t kernel_h, size_t kernel_w, size_t stride_h, size_t stride_w,
            size_t pad_h, size_t pad_w, size_t output_h, size_t output_w);

// Padding/Unpadding operations
template <typename T>
void pad(const T *input, T *output, size_t batch_size, size_t channels, size_t height, size_t width,
         size_t pad_h, size_t pad_w, T value);

template <typename T>
void unpad(const T *input, T *output, size_t batch_size, size_t channels, size_t height,
           size_t width, size_t pad_h, size_t pad_w);

// Crop operation
template <typename T>
void crop(const T *input, T *output, size_t batch_size, size_t channels, size_t height,
          size_t width, size_t start_h, size_t start_w, size_t new_height, size_t new_width);

// Softmax operation
template <typename T>
void softmax(T *data, size_t batch_size, size_t num_classes, size_t height, size_t width);

} // namespace cuda