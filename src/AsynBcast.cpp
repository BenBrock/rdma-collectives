#include <chrono>
#include <cstdio>
#include <cassert>
#include <unistd.h>

#include <upcxx/upcxx.hpp>

template <typename T>
struct broadcast_data {
  broadcast_data(size_t n) {
    bcast_size = n;
    root = 0;
    left = 0;
    right = upcxx::rank_n() - 1;
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

  // only meaningful called after get() is done
  bool futures_done() {
    std::vector<upcxx::future<>> temp;
    for (size_t i = 0; i < futures.size(); i++){
      if (!futures[i].ready()){
        upcxx::progress();
        temp.push_back(futures[i]);
      }
    }  
    bool res = false;
    if (temp.size() == 0)
      res = true;
    futures = temp;
    return res;
  }

  void wait_data(){
    while (check_ready() != true){
      get();
      usleep(100);
    }
  }

  void wait_issue(){
    while (get() != true)
      usleep(100);
  }

  void wait_put(){
    while (futures_done() != true)
      usleep(100);
  }
  
  bool check_ready() {
    if (upcxx::rget(confirmation_ptrs[upcxx::rank_me()]).wait() == 1)
      return true;
    else
      return false;
  }
  
  bool get() { 
    if (left == right)
      return true;
    size_t mid = left + (right - left) / 2;
    size_t dest = (root <= mid) ? right: left;

    if (upcxx::rank_me() == root){ 
      if (!check_ready()) { 
        return false;
      }
      int flag = 1;
      upcxx::future<> fut = upcxx::rput(my_data(), data_ptrs[dest], bcast_size)
      .then([=](){
          return upcxx::rput(&flag, confirmation_ptrs[dest], 1);
        });
      futures.push_back(fut);
    }

    if (upcxx::rank_me() <= mid && root <= mid){
      root = root;
      left = left;
      right = mid;
    }
    else if (upcxx::rank_me() <= mid && root > mid){
      root = dest;
      left = left;
      right = mid;
    }
    else if (upcxx::rank_me() > mid && root <= mid){
      root = dest;
      left = mid+1;
      right = right;
    }
    else if  (upcxx::rank_me() > mid && root > mid){
      root = root;
      left = mid+1;
      right = right;
    }

    return false;
  }

  void init_root(const std::vector<T>& data){
      int flag = 1;
      upcxx::rput(data.data(), data_ptrs[root], data.size()).wait();
      upcxx::rput(&flag, confirmation_ptrs[root], 1).wait();
  }

  T* my_data() {
    return data_ptrs[upcxx::rank_me()].local();
  }

  size_t bcast_size, left, right, root;
  std::vector<upcxx::future<>> futures;
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
    bcast.init_root(data);  
  }
  
  bcast.wait_data();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();
  printf("(1) took %lf seconds until data is available for rank %d.\n", duration, upcxx::rank_me());
  
  bcast.wait_issue();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  printf("(2) took %lf seconds until all rputs issued for rank %d.\n", duration, upcxx::rank_me());
  
  bcast.wait_put();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  printf("(3) took %lf seconds until all work finished for rank %d.\n", duration, upcxx::rank_me());
  

  upcxx::barrier();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();

  if (upcxx::rank_me() == 0) {
    printf("(4) Broadcast took %lf seconds.\n", duration);
  }

  for (size_t i = 0; i < bcast_size; i++) {
    assert(bcast.my_data()[i] == 12);
  }

  upcxx::finalize();
  return 0;
}
