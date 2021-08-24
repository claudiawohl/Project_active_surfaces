add_amdis_executable(
  NAME "project_active_surfaces"
  SOURCES project_active_surfaces.cpp
  DIM 2
  DOW 2
  ALBERTA_GRID)

add_amdis_executable(
        NAME "Sandbox"
        SOURCES test_active_surfaces.cpp
        DIM 2
        DOW 2
        ALBERTA_GRID)

add_amdis_executable(
        NAME "Benchmark"
        SOURCES benchmark.cpp
        DIM 2
        DOW 2
        ALBERTA_GRID)

add_amdis_executable(
        NAME "cahn_hilliard"
        SOURCES cahn_hilliard.cc
        DIM 3
        DOW 3
        ALBERTA_GRID)
