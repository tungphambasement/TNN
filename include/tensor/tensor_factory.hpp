#pragma once

#include "device/device_allocator.hpp"
#include "device/iallocator.hpp"
#include "tensor/tensor.hpp"
#include "tensor/typed_tensor.hpp"
#include "type/type.hpp"

namespace tnn {

template <typename T>
inline Tensor make_tensor(std::vector<size_t> shape, const Device &device = getHost()) {
  auto &allocator = DeviceAllocator::instance(device);
  return std::make_shared<TypedTensor>(allocator, dtype_of<T>(), shape);
}

template <typename T>
inline Tensor make_tensor(std::vector<size_t> shape, const dptr &data,
                          const Device &device = getHost()) {
  auto &allocator = DeviceAllocator::instance(device);
  return std::make_shared<TypedTensor>(allocator, dtype_of<T>(), shape, data);
}

template <typename T>
inline Tensor make_tensor(std::initializer_list<size_t> shape = {},
                          const Device &device = getHost()) {
  auto &allocator = DeviceAllocator::instance(device);
  return std::make_shared<TypedTensor>(allocator, dtype_of<T>(), shape);
}

template <typename T>
inline Tensor make_tensor(std::initializer_list<size_t> shape, const dptr &data,
                          const Device &device = getHost()) {
  auto &allocator = DeviceAllocator::instance(device);
  return std::make_shared<TypedTensor>(allocator, dtype_of<T>(), shape, data);
}

inline Tensor make_tensor(DType_t dtype, std::vector<size_t> shape,
                          const Device &device = getHost()) {
  auto &allocator = DeviceAllocator::instance(device);
  return std::make_shared<TypedTensor>(allocator, dtype, shape);
}

inline Tensor make_tensor(DType_t dtype, std::initializer_list<size_t> shape = {},
                          const Device &device = getHost()) {
  auto &allocator = DeviceAllocator::instance(device);
  return std::make_shared<TypedTensor>(allocator, dtype, shape);
}

template <typename T>
inline Tensor make_tensor(IAllocator &allocator, dptr &&data, std::vector<size_t> shape) {
  return std::make_shared<TypedTensor>(allocator, dtype_of<T>(), std::move(data), shape);
}

inline Tensor make_tensor(IAllocator &allocator, DType_t dtype, dptr &&data,
                          std::vector<size_t> shape) {
  return std::make_shared<TypedTensor>(allocator, dtype, std::move(data), shape);
}

template <typename T>
inline Tensor make_tensor(IAllocator &allocator, std::vector<size_t> shape) {
  return std::make_shared<TypedTensor>(allocator, dtype_of<T>(), shape);
}

template <typename T>
inline Tensor make_tensor(IAllocator &allocator, std::initializer_list<size_t> shape = {}) {
  return std::make_shared<TypedTensor>(allocator, dtype_of<T>(), std::vector<size_t>(shape));
}

inline Tensor make_tensor(IAllocator &allocator, DType_t dtype, std::vector<size_t> shape) {
  return std::make_shared<TypedTensor>(allocator, dtype, shape);
}

inline Tensor make_tensor(IAllocator &allocator, DType_t dtype,
                          std::initializer_list<size_t> shape = {}) {
  return std::make_shared<TypedTensor>(allocator, dtype, shape);
}

inline Tensor create_like(const ConstTensor &other) {
  return make_tensor(other->allocator(), other->data_type(), other->shape());
}

inline Tensor dtype_cast(const ConstTensor &input, DType_t target_dtype) {
  if (input->data_type() == target_dtype) {
    return input->clone();
  }

  const dptr &input_data = input->data_ptr();
  size_t input_size = input->size();
  dptr output_data = make_dptr(input->device(), input_size * get_dtype_size(target_dtype));
  DISPATCH_ON_ANY_DTYPE(
      input->data_type(), A_T,
      DISPATCH_DTYPE(target_dtype, B_T, ops::cast<A_T, B_T>(input_data, output_data, input_size)));
  return make_tensor(input->allocator(), target_dtype, std::move(output_data), input->shape());
}

inline Tensor load(std::ifstream &in, IAllocator &allocator) {
  if (!in.is_open()) {
    throw std::runtime_error("File is not open for reading");
  }
  DType_t dtype;
  in.read(reinterpret_cast<char *>(&dtype), sizeof(DType_t));
  size_t dims;
  in.read(reinterpret_cast<char *>(&dims), sizeof(size_t));
  std::vector<size_t> shape(dims);
  in.read(reinterpret_cast<char *>(shape.data()), dims * sizeof(size_t));
  if (in.gcount() != static_cast<std::streamsize>(dims * sizeof(size_t))) {
    throw std::runtime_error("Failed to read tensor shape from file");
  }
  auto tensor = make_tensor(allocator, dtype, shape);
  size_t byte_size = tensor->size() * get_dtype_size(dtype);
  if (allocator.device().device_type() == DeviceType::CPU) {
    in.read(reinterpret_cast<char *>(tensor->data()), byte_size);
    if (in.gcount() != static_cast<std::streamsize>(byte_size)) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
  } else {
    std::vector<char> host_buffer(byte_size);
    in.read(reinterpret_cast<char *>(host_buffer.data()), byte_size);
    if (in.gcount() != static_cast<std::streamsize>(byte_size)) {
      throw std::runtime_error("Failed to read tensor data from file");
    }
    allocator.device().copyToDevice(tensor->data(), host_buffer.data(), byte_size);
  }
  return tensor;
}

}  // namespace tnn