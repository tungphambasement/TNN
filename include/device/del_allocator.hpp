#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "device/dptr.hpp"
#include "device/flow.hpp"
#include "device/iallocator.hpp"

namespace tnn {

inline size_t align_up(size_t size, size_t alignment = DEFAULT_ALIGNMENT) {
  return (size + alignment - 1) & ~(alignment - 1);
}

// Double-ended linear allocator that maintains a single contiguous buffer and allocates from both
// ends depending on the side variable
class DELAllocator : public IAllocator {
public:
  DELAllocator(const Device &device, flowHandle_t flow = defaultFlowHandle)
      : device_(device),
        flow_(flow),
        slab_ptr_(nullptr),
        capacity_(0),
        left_offset_(0),
        right_offset_(0),
        side_(0) {}

  ~DELAllocator() {
    clear();
    std::lock_guard<std::mutex> lock(mutex_);
    if (slab_ptr_) {
      device_.deallocateAlignedMemory(slab_ptr_);
      slab_ptr_ = nullptr;
    }
  }

  // Delete copy semantics
  DELAllocator(const DELAllocator &) = delete;
  DELAllocator &operator=(const DELAllocator &) = delete;

  dptr allocate(size_t size) override {
    if (size == 0) return dptr(nullptr);

    size_t aligned_size = align_up(size);

    std::lock_guard<std::mutex> lock(mutex_);

    if (!slab_ptr_) {
      throw std::runtime_error("DELAllocator: Master slab not initialized. Call reserve() first.");
    }

    // Try finding in free blocks first (best fit)
    auto it = free_blocks_by_size_.lower_bound(aligned_size);
    if (it != free_blocks_by_size_.end()) {
      size_t block_size = it->first;
      size_t offset = it->second;

      free_blocks_by_size_.erase(it);
      free_blocks_by_offset_.erase(offset);

      // If the block is larger than requested, split it and return the remainder
      if (block_size > aligned_size) {
        size_t new_offset = offset + aligned_size;
        size_t new_size = block_size - aligned_size;
        free_blocks_by_offset_.emplace(new_offset, new_size);
        free_blocks_by_size_.emplace(new_size, new_offset);
      }

      return create_dptr(offset, aligned_size);
    }

    // Allocate from slab ends (bump pointer)
    size_t offset = 0;
    if (side_ == 0) {  // left
      if (left_offset_ + aligned_size > right_offset_) {
        throw std::runtime_error("DELAllocator: Out of memory in master slab");
      }
      offset = left_offset_;
      left_offset_ += aligned_size;
    } else {  // right
      if (right_offset_ < left_offset_ + aligned_size) {
        throw std::runtime_error("DELAllocator: Out of memory in master slab");
      }
      right_offset_ -= aligned_size;
      offset = right_offset_;
    }

    return create_dptr(offset, aligned_size);
  }

  void flip() {
    std::lock_guard<std::mutex> lock(mutex_);
    side_ = 1 - side_;
  }

  void clear() override {
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_by_offset_.clear();
    free_blocks_by_size_.clear();
    left_offset_ = 0;
    right_offset_ = capacity_;
  }

  void reserve(size_t size) override {
    std::lock_guard<std::mutex> lock(mutex_);
    if (slab_ptr_) {
      throw std::runtime_error("DELAllocator: Master slab already reserved");
    }

    if (size == 0) {
      return;
    }

    capacity_ = size;
    slab_ptr_ = device_.allocateAlignedMemory(capacity_, DEFAULT_ALIGNMENT);
    if (!slab_ptr_) {
      throw std::runtime_error("DELAllocator: Failed to allocate master slab");
    }

    left_offset_ = 0;
    right_offset_ = capacity_;
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_blocks_by_size_.size();
  }

  size_t cached_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto &pair : free_blocks_by_size_) {
      total += pair.first;
    }
    return total;
  }

  const Device &device() const override { return device_; }

private:
  const Device &device_;
  flowHandle_t flow_;
  mutable std::mutex mutex_;
  void *slab_ptr_;
  size_t capacity_;
  size_t left_offset_;
  size_t right_offset_;
  int side_;

  std::map<size_t, size_t> free_blocks_by_offset_;     // offset -> size
  std::multimap<size_t, size_t> free_blocks_by_size_;  // size -> offset

  dptr create_dptr(size_t offset, size_t size) {
    void *slice_ptr = static_cast<uint8_t *>(slab_ptr_) + offset;

    auto storage = std::shared_ptr<device_storage>(
        new device_storage(device_, slice_ptr, size, DEFAULT_ALIGNMENT),
        [this, offset, size](device_storage *storage) {
          this->reclaim(offset, size);
          delete storage;
        });

    return dptr(storage, 0, size);
  }

  void reclaim(size_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t new_offset = offset;
    size_t new_size = size;

    // Coalesce with adjacent free blocks
    auto next_it = free_blocks_by_offset_.upper_bound(offset);

    // Check with the previous block
    if (next_it != free_blocks_by_offset_.begin()) {
      auto prev_it = std::prev(next_it);
      if (prev_it->first + prev_it->second == new_offset) {
        new_offset = prev_it->first;
        new_size += prev_it->second;

        // Remove prev from size map
        remove_from_size_map(prev_it->second, prev_it->first);
        free_blocks_by_offset_.erase(prev_it);
      }
    }

    // Check with the next block
    if (next_it != free_blocks_by_offset_.end()) {
      if (new_offset + new_size == next_it->first) {
        new_size += next_it->second;

        // Remove next from size map
        remove_from_size_map(next_it->second, next_it->first);
        next_it = free_blocks_by_offset_.erase(next_it);
      }
    }

    // Shrink allocated boundaries if touching the bump pointers
    if (new_offset + new_size == left_offset_) {
      left_offset_ = new_offset;
      return;
    }

    if (new_offset == right_offset_) {
      right_offset_ = new_offset + new_size;
      return;
    }

    // Insert new coalesced block (only if it doesn't touch the edges)
    free_blocks_by_offset_.emplace(new_offset, new_size);
    free_blocks_by_size_.emplace(new_size, new_offset);
  }

  void remove_from_size_map(size_t size, size_t offset) {
    auto range = free_blocks_by_size_.equal_range(size);
    for (auto it = range.first; it != range.second; ++it) {
      if (it->second == offset) {
        free_blocks_by_size_.erase(it);
        break;
      }
    }
  }
};
}  // namespace tnn
