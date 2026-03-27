cmake_minimum_required(VERSION 3.23)

project(cudaSift
  VERSION 2.0.0
  LANGUAGES C CXX CUDA
)

# C11, C++17 standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Options ──────────────────────────────────────────────
option(CUSIFT_BUILD_SHARED   "Build cusift as a shared library instead of static" OFF)
option(CUSIFT_USE_FAST_MATH  "Compile CUDA code with --use_fast_math"             OFF)
option(CUSIFT_ALL_CUDA_ARCHS "Build for all supported CUDA architectures"         OFF)

# ── OpenMP ───────────────────────────────────────────────
find_package(OpenMP COMPONENTS C CXX)

# ── CUDA architecture ───────────────────────────────────
# Override with -DCMAKE_CUDA_ARCHITECTURES="75;86" etc.
if(CUSIFT_ALL_CUDA_ARCHS)
  set(CMAKE_CUDA_ARCHITECTURES all)
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

# ── Source files ──────────────────────────────────────────
set(CUDA_SOURCES
  cudaImage.cu
  cudaSiftH.cu
  matching.cu
  cusift.cu
)

set(C_SOURCES
  geomFuncs.c
)

# ── CUDA object library (compiled by nvcc) ───────────────
add_library(cusift_cuda OBJECT ${CUDA_SOURCES})
target_include_directories(cusift_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(cusift_cuda PRIVATE VERBOSE)

if(MSVC)
  target_compile_options(cusift_cuda PRIVATE
    -O2 --Werror cross-execution-space-call
    -Xcompiler=/W4 -Xcompiler=/Zc:preprocessor
  )
else()
  target_compile_options(cusift_cuda PRIVATE
    -O2 --Werror cross-execution-space-call
    -Xcompiler=-Wall,-Wextra
  )
endif()

if(CUSIFT_USE_FAST_MATH)
  target_compile_options(cusift_cuda PRIVATE --use_fast_math)
endif()

# ── cusift library (C/C++ sources + CUDA objects) ────────
if(CUSIFT_BUILD_SHARED)
  set_target_properties(cusift_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_definitions(cusift_cuda PRIVATE CUSIFT_EXPORTS)
  add_library(cusift SHARED ${C_SOURCES} $<TARGET_OBJECTS:cusift_cuda>)
  target_compile_definitions(cusift PRIVATE CUSIFT_EXPORTS)
  set_target_properties(cusift PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
else()
  add_library(cusift STATIC ${C_SOURCES} $<TARGET_OBJECTS:cusift_cuda>)
  target_compile_definitions(cusift PUBLIC CUSIFT_STATIC)
endif()

target_include_directories(cusift
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(cusift PRIVATE VERBOSE)

# C/C++ compile options (no COMPILE_LANGUAGE guards needed — only C sources in this target)
if(MSVC)
  target_compile_options(cusift PRIVATE /W4 /O2 /Zc:preprocessor /D_CRT_SECURE_NO_WARNINGS)
  set_source_files_properties(geomFuncs.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  target_compile_options(cusift PRIVATE -Wall -Wextra -Wpedantic -O2)
  set_source_files_properties(geomFuncs.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# Link CUDA runtime (needed since cusift itself has no .cu files)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cusift PRIVATE CUDA::cudart_static)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
  target_link_libraries(cusift PRIVATE OpenMP::OpenMP_CXX)
  if(NOT MSVC AND OpenMP_CXX_FLAGS)
    target_compile_options(cusift_cuda PRIVATE -Xcompiler=${OpenMP_CXX_FLAGS})
  endif()
endif()

# ── Demo executable ──────────────────────────────────────
add_executable(cusift_demo test/main.cpp)
target_include_directories(cusift_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
target_link_libraries(cusift_demo PRIVATE cusift)

# ── Speed benchmark executable ──────────────────────────
add_executable(cusift_speed test/speed.cpp)
target_include_directories(cusift_speed PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
target_link_libraries(cusift_speed PRIVATE cusift)

# ── Install ──────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS cusift
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS cusift_demo cusift_speed
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES cusift.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY test/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cusift/test
  FILES_MATCHING
    PATTERN "*.png"
    PATTERN "*.py"
    PATTERN "*.cpp"
)

# ── CPack ────────────────────────────────────────────────
set(CPACK_GENERATOR "ZIP")
include(CPack)
