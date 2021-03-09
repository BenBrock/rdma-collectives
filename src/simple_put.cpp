#include <chrono>
#include <cstdio>
#include <cassert>

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

  // Broadcast vector `data` from process `root` to
  // all other processes.
  void broadcast_simple(const std::vector<T>& data, size_t root) {
    if (upcxx::rank_me() == root) {
      for (size_t i = 0; i < upcxx::rank_n(); i++) {
        int flag = 1;
        upcxx::rput(data.data(), data_ptrs[i], data.size()).wait();
        upcxx::rput(&flag, confirmation_ptrs[i], 1).wait();
      }
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

  size_t bcast_size = 100;

  // Initialize a broadcast "data structure"
  // to support broadcasts up to `bcast_size` ints.
  
  broadcast_data<int> bcast(bcast_size);

  upcxx::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  if (upcxx::rank_me() == 0) {
    std::vector<int> data(bcast_size, 12);
    bcast.broadcast_simple(data, 0);
  }

  while (!bcast.check_ready()) {
  }
  upcxx::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  if (upcxx::rank_me() == 0) {
    printf("Broadcast took %lf seconds.\n", duration);
  }

  for (size_t i = 0; i < bcast_size; i++) {
    assert(bcast.my_data()[i] == 12);
  }

  upcxx::finalize();
  return 0;
}
