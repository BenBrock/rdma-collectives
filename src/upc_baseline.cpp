#include <chrono>
#include <cstdio>
#include <cassert>
#include <unistd.h>
#include <upcxx/upcxx.hpp>

template <typename T>
struct broadcast_data {
  broadcast_data(size_t n) {
    for (size_t i = 0; i < upcxx::rank_n(); i++) {
      upcxx::global_ptr<T> ptr = nullptr;
      if (upcxx::rank_me() == i) {
        ptr = upcxx::new_array<T>(n);
      }
      ptr = upcxx::broadcast(ptr, i).wait();
      data_ptrs.push_back(ptr);

      upcxx::global_ptr<int> cptr = nullptr;
      if (upcxx::rank_me() == i) {
        cptr = upcxx::new_array<int>(1);
        *cptr.local() = 0;
      }
      cptr = upcxx::broadcast(cptr, i).wait();
      confirmation_ptrs.push_back(cptr);
    }
  }


  bool check_ready() {
    if (upcxx::rget(confirmation_ptrs[upcxx::rank_me()]).wait() == 1) {
      return true;
    } else {
      return false;
    }
  }

  T* my_data() {
    return data_ptrs[upcxx::rank_me()].local();
  }

  // Global pointers to data buffer for each process.
  std::vector<upcxx::global_ptr<T>> data_ptrs;
  // Global pointers to confirmation flag for each process.
  std::vector<upcxx::global_ptr<int>> confirmation_ptrs;
};
int main(int argc, char** argv) {
  upcxx::init();

  size_t bcast_size = 1000000;

  // Initialize a broadcast "data structure"
  // to support broadcasts up to `bcast_size` ints.
  
  broadcast_data<int> bcast(bcast_size);
  std::vector<int> data(bcast_size, 0);

  if (upcxx::rank_me() == 0) {
    for(int i = 0; i < data.size(); i ++){
        data[i] = 12;
    }
  }

  upcxx::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  upcxx::broadcast(data.data(), bcast_size, 0).wait();

  // upcxx::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  printf("Rank %d: %lf \n", upcxx::rank_me(), duration);
  


  for (size_t i = 0; i < bcast_size; i++) {
    assert(data[i] == 12);
  }

  upcxx::finalize();
  return 0;
}
