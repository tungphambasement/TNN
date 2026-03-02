#include <cassert>

#include "device/del_allocator.hpp"
#include "device/device_manager.hpp"

using namespace tnn;

int main() {
  DELAllocator alloc(getHost(), defaultFlowHandle);
  alloc.reserve(100);

  auto ptr1 = alloc.allocate(10);
  auto ptr2 = alloc.allocate(20);
  alloc.flip();
  auto ptr3 = alloc.allocate(15);
  auto ptr4 = alloc.allocate(25);

  // ptr1 [0-10]
  // ptr2 [10-30]
  // ptr3 [85-100]
  // ptr4 [60-85]

  // unallocated [30-60]

  // Now we force them out of scope and ensure they are reclaimed
  ptr2 = nullptr;

  ptr3 = nullptr;

  ptr4 = nullptr;

  auto ptrX = alloc.allocate(30);

  return 0;
}
