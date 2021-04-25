#include <mpi.h>
#include <chrono>
#include <cstdio>
#include <cassert>
#include <vector>
#include <unistd.h>

int main(int argc, char** argv) {
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
  double duration = std::chrono::duration<double>(end - begin).count();
  printf("Rank \t %d: \t %lf \n", rank, duration);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("Broadcast took \t %lf \t seconds.\n", duration);
  }
  
  for (size_t i = 0; i < bcast_size; i++) {
    assert(data[i] == 12);
  }

  return 0;
}
