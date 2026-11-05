cmake_minimum_required(VERSION 3.23)

# ── CUDA architecture ───────────────────────────────────
# This must run BEFORE project(): enabling the CUDA language inside project()
# defines CMAKE_CUDA_ARCHITECTURES from nvcc's own default (a single
# architecture), after which a "NOT DEFINED" check never fires and the project
# default is silently ignored.  A -DCMAKE_CUDA_ARCHITECTURES=... on the command
# line is already defined here and wins.
option(CUSIFT_ALL_CUDA_ARCHS "Build for all supported CUDA architectures" OFF)
if(CUSIFT_ALL_CUDA_ARCHS)
  set(CMAKE_CUDA_ARCHITECTURES all)
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  # One cubin per major architecture the toolkit supports, plus PTX for the
  # newest.  A cubin built for X.0 runs on every X.y, so this covers every GPU
  # the toolkit can drive, Jetson Orin (8.7) included, at a fraction of the
  # build time of "all".  Pass -DCMAKE_CUDA_ARCHITECTURES=87 (or 72 for
  # Xavier under CUDA 11/12) for a build tuned to one board, or
  # -DCUSIFT_ALL_CUDA_ARCHS=ON for a cubin per minor architecture.
  set(CMAKE_CUDA_ARCHITECTURES all-major)
endif()

project(cudaSift
  VERSION 2.4.0
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
option(CUSIFT_ENABLE_AVX2    "Compile geomFuncs.c with AVX2 + FMA (x86/x64 only)" ON)
# CUSIFT_ALL_CUDA_ARCHS is declared above project(), where it has to be.

# Detect whether the host architecture even supports AVX2 / FMA. On non-x86
# targets (e.g. arm64) the AVX2 path is unconditionally skipped, regardless
# of CUSIFT_ENABLE_AVX2, and the C source falls back to its scalar path via
# the preprocessor guards in src/geomFuncs.c.
set(_CUSIFT_X86 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64|i[3-6]86|x86)")
  set(_CUSIFT_X86 TRUE)
endif()

# ── OpenMP ───────────────────────────────────────────────
find_package(OpenMP COMPONENTS C CXX)

message(STATUS "cusift: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")

# ── Source files ──────────────────────────────────────────
set(CUDA_SOURCES
  src/cudaImage.cu
  src/cudaSiftH.cu
  src/matching.cu
  src/cusift.cu
)

set(C_SOURCES
  src/geomFuncs.c
)

# ── CUDA object library (compiled by nvcc) ───────────────
add_library(cusift_cuda OBJECT ${CUDA_SOURCES})
target_include_directories(cusift_cuda PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
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
  # The CUDA objects include cusift.h too; without the define they see the
  # API declared dllimport while defining it (nvcc warning #1390-D on MSVC).
  target_compile_definitions(cusift_cuda PRIVATE CUSIFT_STATIC)
endif()

target_include_directories(cusift
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(cusift PRIVATE VERBOSE)

# C/C++ compile options (no COMPILE_LANGUAGE guards needed — only C sources in this target)
if(MSVC)
  target_compile_options(cusift PRIVATE /W4 /O2 /Zc:preprocessor /D_CRT_SECURE_NO_WARNINGS)
  if(CUSIFT_ENABLE_AVX2 AND _CUSIFT_X86)
    set_source_files_properties(src/geomFuncs.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    message(STATUS "cusift: building geomFuncs.c with AVX2 + FMA (/arch:AVX2)")
  else()
    message(STATUS "cusift: building geomFuncs.c with scalar fallback (no AVX2)")
  endif()
else()
  target_compile_options(cusift PRIVATE -Wall -Wextra -Wpedantic -O2)
  if(CUSIFT_ENABLE_AVX2 AND _CUSIFT_X86)
    set_source_files_properties(src/geomFuncs.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    message(STATUS "cusift: building geomFuncs.c with AVX2 + FMA (-mavx2 -mfma)")
  else()
    message(STATUS "cusift: building geomFuncs.c with scalar fallback (no AVX2)")
  endif()
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
include(CMakePackageConfigHelpers)

install(TARGETS cusift
  EXPORT cusiftTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(TARGETS cusift_demo cusift_speed
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES include/cusift.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY test/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/cusift/test
  FILES_MATCHING
    PATTERN "*.png"
    PATTERN "*.py"
    PATTERN "*.cpp"
)

# ── Package config (find_package(cusift)) ────────────────
set(CUSIFT_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/cusift")

install(EXPORT cusiftTargets
  FILE        cusiftTargets.cmake
  NAMESPACE   cusift::
  DESTINATION ${CUSIFT_INSTALL_CMAKEDIR}
)

# Generated alongside the build tree so consumers using add_subdirectory()
# can also resolve cusift::cusift identically to the install path.
add_library(cusift::cusift ALIAS cusift)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cusiftConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/cusiftConfig.cmake"
  INSTALL_DESTINATION ${CUSIFT_INSTALL_CMAKEDIR}
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/cusiftConfigVersion.cmake"
  VERSION       ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/cusiftConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/cusiftConfigVersion.cmake"
  DESTINATION ${CUSIFT_INSTALL_CMAKEDIR}
)

# ── CPack ────────────────────────────────────────────────
set(CPACK_GENERATOR "ZIP")
include(CPack)
