cmake_minimum_required(VERSION 3.14)
project(final)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MPI REQUIRED)

add_executable(mpi_baseline MPI_baseline.cpp)
target_link_libraries(mpi_baseline PRIVATE MPI::MPI_CXX)

# Copy the job scripts
configure_file(job-mpi-put job-mpi-put COPYONLY)