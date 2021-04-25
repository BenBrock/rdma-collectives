#include <chrono>
#include <cstdio>
#include <cassert>

#include <upcxx/upcxx.hpp>

template <typename T>
struct broadcast_data {
  broadcast_data(size_t n) {
    bcast_size = n;
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
  void broadcast_MST(size_t root, size_t left, size_t right) { 
	  if (left == right)
      return;
    size_t mid = left + (right - left) / 2;
    size_t dest = (root <= mid) ? right: left;

    if (upcxx::rank_me() == root){ 
      while (!check_ready()) { 
        // incase we send before previous recursion has not finished putting
      }
      const T* data = my_data();
      int flag = 1;
      upcxx::rput(data, data_ptrs[dest], bcast_size).wait();
      upcxx::rput(&flag, confirmation_ptrs[dest], 1).wait();
    }

    if (upcxx::rank_me() <= mid && root <= mid)
      broadcast_MST(root, left, mid);
    else if (upcxx::rank_me() <= mid && root > mid)
      broadcast_MST(dest, left, mid);
    else if (upcxx::rank_me() > mid && root <= mid)
      broadcast_MST(dest, mid+1, right);
    else if  (upcxx::rank_me() > mid && root > mid)
      broadcast_MST(root, mid+1, right);
  }

  bool check_ready() {
    if (upcxx::rget(confirmation_ptrs[upcxx::rank_me()]).wait() == 1) {
      return true;
    } else {
      return false;
    }
  }

  void init_root(const std::vector<T>& data, size_t root){
    if (upcxx::rank_me() == root) {
      int flag = 1;
      upcxx::rput(data.data(), data_ptrs[root], data.size()).wait();
      upcxx::rput(&flag, confirmation_ptrs[root], 1).wait();
    }
  }

  T* my_data() {
    return data_ptrs[upcxx::rank_me()].local();
  }

  size_t bcast_size;
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

  upcxx::barrier();
  auto begin = std::chrono::high_resolution_clock::now();
  
  if (upcxx::rank_me() == 0) {
    std::vector<int> data(bcast_size, 12);
    bcast.init_root(data, 0);
  }

  bcast.broadcast_MST(0, 0, upcxx::rank_n()-1);

  while (!bcast.check_ready()) {
  }

  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  printf("Rank \t %d \t %lf \t \n", upcxx::rank_me(), duration);

  upcxx::barrier();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();

  if (upcxx::rank_me() == 0) {
    printf("Broadcast took %lf seconds.\n", duration);
  }

  for (size_t i = 0; i < bcast_size; i++) {
    assert(bcast.my_data()[i] == 12);
  }

  upcxx::finalize();
  return 0;
}
