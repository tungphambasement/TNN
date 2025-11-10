#include "device/cuda/cuda_context.hpp"

#ifdef USE_CUDA

#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tnn {

CUDAContext::CUDAContext(int id) : Context() {
  // Set the device for this context
  cudaError_t err = cudaSetDevice(id);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to set CUDA device " + std::to_string(id) + ": " +
                             cudaGetErrorString(err));
  }
}

size_t CUDAContext::getTotalMemory() const {
  size_t total_mem = 0;
  cudaError_t err = cudaMemGetInfo(nullptr, &total_mem);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to get total CUDA memory: " +
                             std::string(cudaGetErrorString(err)));
  }
  return total_mem;
}

size_t CUDAContext::getAvailableMemory() const {
  size_t free_mem = 0;
  cudaError_t err = cudaMemGetInfo(&free_mem, nullptr);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to get available CUDA memory: " +
                             std::string(cudaGetErrorString(err)));
  }
  return free_mem;
}

void *CUDAContext::allocateMemory(size_t size) {
  void *ptr = nullptr;
  cudaError_t err = cudaMalloc(&ptr, size);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate CUDA memory: " +
                             std::string(cudaGetErrorString(err)));
  }
  return ptr;
}

void CUDAContext::deallocateMemory(void *ptr) {
  if (ptr != nullptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
      throw std::runtime_error("Failed to free CUDA memory: " +
                               std::string(cudaGetErrorString(err)));
    }
  }
}

void CUDAContext::copyToDevice(void *dest, const void *src, size_t size) {
  cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to copy memory to CUDA device: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void CUDAContext::copyToHost(void *dest, const void *src, size_t size) {
  cudaError_t err = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to copy memory from CUDA device: " +
                             std::string(cudaGetErrorString(err)));
  }
}

void *CUDAContext::allocateAlignedMemory(size_t size, size_t alignment) {
  if (alignment <= 256) {
    return allocateMemory(size);
  }

  void *raw_ptr = nullptr;
  size_t total_size = size + alignment;
  cudaError_t err = cudaMalloc(&raw_ptr, total_size);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate aligned CUDA memory: " +
                             std::string(cudaGetErrorString(err)));
  }

  uintptr_t addr = reinterpret_cast<uintptr_t>(raw_ptr);
  uintptr_t aligned_addr = ((addr + alignment - 1) / alignment) * alignment;

  uintptr_t offset = aligned_addr - addr;
  if (offset < sizeof(void *)) {
    aligned_addr += alignment;
    offset = aligned_addr - addr;
  }

  void **metadata = reinterpret_cast<void **>(aligned_addr - sizeof(void *));
  cudaMemcpy(metadata, &raw_ptr, sizeof(void *), cudaMemcpyHostToDevice);

  return reinterpret_cast<void *>(aligned_addr);
}

void CUDAContext::deallocateAlignedMemory(void *ptr) {
  if (ptr != nullptr) {
    void *original_ptr = nullptr;
    void **metadata = reinterpret_cast<void **>(reinterpret_cast<uintptr_t>(ptr) - sizeof(void *));

    cudaError_t err = cudaMemcpy(&original_ptr, metadata, sizeof(void *), cudaMemcpyDeviceToHost);
    if (err == cudaSuccess && original_ptr != nullptr) {
      cudaFree(original_ptr);
    } else {
      cudaFree(ptr);
    }
  }
}

} // namespace tnn

#endif // USE_CUDA