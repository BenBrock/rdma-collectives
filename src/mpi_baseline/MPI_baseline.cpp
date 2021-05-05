#include <mpi.h>
#include <chrono>
#include <cstdio>
#include <cassert>
#include <vector>
#include <unistd.h>
#include <string.h>

int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], option) == 0) {
            return i;
        }
    }
    return -1;
}

bool find_int_arg(int argc, char** argv, const char* option, bool default_value) {
    int iplace = find_arg_idx(argc, argv, option);

    if (iplace >= 0 && iplace < argc) {
        return true;
    }

    return default_value;
}

int main(int argc, char** argv) {
  bool kernel = find_int_arg(argc, argv, "-k", false);
  int num_procs, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t bcast_size = 1000000;
  std::vector<int> data(bcast_size, 0);
  if (rank == 0) {
    for(int i = 0; i < data.size(); i ++){
        data[i] = 12;
    }
  }

  auto begin = std::chrono::high_resolution_clock::now();

  MPI_Bcast(data.data(), bcast_size, MPI_INT, 0, MPI_COMM_WORLD);
  auto end = std::chrono::high_resolution_clock::now();
  double duration_data = std::chrono::duration<double>(end - begin).count();

  double duration_kernel = 0;
  if (kernel){
    usleep(500000);
    end = std::chrono::high_resolution_clock::now();
    duration_kernel = std::chrono::duration<double>(end - begin).count();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double total_duration_data = 0;
  MPI_Reduce(&duration_data, &total_duration_data, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  double total_duration_kernel = 0;
  MPI_Reduce(&duration_kernel, &total_duration_kernel, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();
  if (rank == 0) {
    printf("(1) \t Data received in \t %lf \t seconds in average.\n", total_duration_data / num_procs);
    printf("(2) \t Kernel done in \t %lf \t seconds in average.\n", total_duration_kernel / num_procs);
    printf("(3) Broadcast took \t %lf \t seconds.\n", duration);
  }
  
  for (size_t i = 0; i < bcast_size; i++) {
    assert(data[i] == 12);
  }

  return 0;
}
