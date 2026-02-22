/*
 * Copyright (c) 2025 Tung D. Pham
 *
 * This software is licensed under the MIT License. See the LICENSE file in the
 * project root for the full license text.
 */
#pragma once

#include "device/device.hpp"
#include "device/dptr.hpp"

namespace tnn {

// Allocators should return a dptr whose internal storage is reclaimed by the allocator itself
class IAllocator {
public:
  virtual ~IAllocator() = default;

  // allocate a ptr with capacity == size
  virtual dptr allocate(size_t size) = 0;

  virtual const Device &device() const = 0;
};
}  // namespace tnn