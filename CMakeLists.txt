cmake_minimum_required(VERSION 3.23)

# ── CUDA architecture ───────────────────────────────────
# This must run BEFORE project(): enabling the CUDA language inside project()
# defines CMAKE_CUDA_ARCHITECTURES from nvcc's own default (a single
# architecture) and caches it.  Because the cache entry then exists on every
# later configure, a "NOT DEFINED CMAKE_CUDA_ARCHITECTURES" guard only works
# for a brand-new build directory; an existing one keeps whatever it cached
# first.  So the project owns the choice through CUSIFT_CUDA_ARCHITECTURES
# and writes CMAKE_CUDA_ARCHITECTURES from it on every configure.
#
# The default is one cubin per major architecture the toolkit supports, plus
# PTX for the newest.  A cubin built for X.0 runs on every X.y, so this
# covers every GPU the toolkit can drive, Jetson Orin (8.7) included, at a
# fraction of the build time of "all".  Pass -DCUSIFT_CUDA_ARCHITECTURES=87
# (or 72 for Xavier under CUDA 11/12) for a build tuned to one board, or
# -DCUSIFT_ALL_CUDA_ARCHS=ON for a cubin per minor architecture.
option(CUSIFT_ALL_CUDA_ARCHS "Build for all supported CUDA architectures" OFF)
set(CUSIFT_CUDA_ARCHITECTURES "all-major" CACHE STRING
  "CUDA architectures to build for, in CMAKE_CUDA_ARCHITECTURES syntax (all-major, all, native, or a list such as 87)")
if(CUSIFT_ALL_CUDA_ARCHS)
  set(_cusift_archs all)
else()
  set(_cusift_archs "${CUSIFT_CUDA_ARCHITECTURES}")
endif()
if(DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES} AND NOT "$CACHE{CMAKE_CUDA_ARCHITECTURES}" STREQUAL "${_cusift_archs}")
  message(STATUS "cusift: replacing cached CMAKE_CUDA_ARCHITECTURES=$CACHE{CMAKE_CUDA_ARCHITECTURES} with ${_cusift_archs}; set CUSIFT_CUDA_ARCHITECTURES to change it")
endif()
set(CMAKE_CUDA_ARCHITECTURES "${_cusift_archs}" CACHE STRING "CUDA architectures (derived from CUSIFT_CUDA_ARCHITECTURES)" FORCE)
unset(_cusift_archs)

project(cudaSift
  VERSION 3.0.0
  LANGUAGES C CXX CUDA
)

# The header carries the version as macros for callers; keep the two in step.
file(STRINGS include/cusift.h _cusift_ver_lines REGEX "#define CUSIFT_VERSION_(MAJOR|MINOR|PATCH)")
string(REGEX REPLACE ".*MAJOR ([0-9]+).*MINOR ([0-9]+).*PATCH ([0-9]+).*" "\\1.\\2.\\3" _cusift_header_version "${_cusift_ver_lines}")
if(NOT _cusift_header_version STREQUAL PROJECT_VERSION)
  message(FATAL_ERROR "include/cusift.h declares version ${_cusift_header_version} but CMakeLists.txt says ${PROJECT_VERSION}; update one of them")
endif()

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
option(CUSIFT_ENABLE_OPENMP  "Thread the host-side warp with OpenMP (adds a dependency on the OpenMP runtime library)" ON)
option(CUSIFT_BUILD_MATLAB   "Build cusift_matlab, the loadlibrary-friendly shared library for MATLAB (see matlab/README.md)" OFF)
option(CUSIFT_STATIC_RUNTIME "Link the C and C++ runtimes statically into the shared libraries, so they load into a process that carries its own, such as MATLAB" OFF)
# CUSIFT_ALL_CUDA_ARCHS and CUSIFT_CUDA_ARCHITECTURES are declared above
# project(), where they have to be.

# A self-contained shared library: no msvcp140.dll / libstdc++.so.6 import.
# The public API is C, no C++ object or exception crosses it, and the library
# frees everything it allocates, so a private copy of the C++ runtime inside
# the library is safe.  On MSVC this is the /MT CRT, applied to C, C++ and the
# nvcc host passes alike (policy CMP0091); the GCC/Clang half is applied per
# target by cusift_self_contained() below.  The OpenMP runtime cannot be
# linked statically on MSVC; CUSIFT_ENABLE_OPENMP=OFF removes that import.
if(CUSIFT_STATIC_RUNTIME AND MSVC)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
if(CUSIFT_STATIC_RUNTIME AND NOT MSVC)
  # Probe the toolchain once: -static-libstdc++ needs libstdc++.a, which
  # Debian/Ubuntu ship with g++ but RHEL-family systems keep in a separate
  # libstdc++-static package that a locked-down machine may not have.
  # Without it the GCC driver turns the flag into -Bstatic -lstdc++ and the
  # build dies late with "cannot find -lstdc++"; better to stop here and say
  # what the alternatives are.  (libgcc.a comes with the compiler itself.)
  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_LINK_OPTIONS -static-libstdc++ -static-libgcc)
  check_cxx_source_compiles("
#include <iostream>
#include <string>
int main() { std::string s(\"static runtime\"); std::cout << s << std::endl; return 0; }
" CUSIFT_HAVE_STATIC_RUNTIME)
  unset(CMAKE_REQUIRED_LINK_OPTIONS)
  if(NOT CUSIFT_HAVE_STATIC_RUNTIME)
    message(FATAL_ERROR
      "CUSIFT_STATIC_RUNTIME=ON, but this toolchain cannot link libstdc++ statically "
      "(no libstdc++.a; the CMake configure log has the linker's output). Either install "
      "it (the libstdc++-static package on RHEL-family systems; it comes with g++ on "
      "Debian/Ubuntu) and delete CUSIFT_HAVE_STATIC_RUNTIME from the cache, or configure "
      "with -DCUSIFT_STATIC_RUNTIME=OFF and build with the GCC version your MATLAB "
      "release supports, so that the libstdc++ MATLAB bundles satisfies the library "
      "(see matlab/README.md).")
  endif()
endif()
function(cusift_self_contained target)
  if(CUSIFT_STATIC_RUNTIME AND NOT MSVC)
    # Link with the C++ driver so CMake does not add its own explicit
    # -lstdc++ (which would bypass -static-libstdc++) for the CUDA objects.
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CXX)
    target_link_options(${target} PRIVATE -static-libstdc++ -static-libgcc)
    if(NOT APPLE)
      # Keep the folded-in libstdc++ and cudart symbols out of the export
      # table, so they cannot interpose on the host process's own copies.
      target_link_options(${target} PRIVATE "LINKER:--exclude-libs,ALL")
    endif()
  endif()
endfunction()

# Position-independent code for every target, static library included, so a
# Linux consumer can link libcusift.a into a shared object (a plugin or a
# Python extension) without "recompile with -fPIC" link errors.  Shared
# libraries get it anyway; on MSVC it is a no-op.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Detect whether the host architecture even supports AVX2 / FMA. On non-x86
# targets (e.g. arm64) the AVX2 path is unconditionally skipped, regardless
# of CUSIFT_ENABLE_AVX2, and the C source falls back to its scalar path via
# the preprocessor guards in src/geomFuncs.c.
set(_CUSIFT_X86 FALSE)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64|amd64|i[3-6]86|x86)")
  set(_CUSIFT_X86 TRUE)
endif()

# ── OpenMP ───────────────────────────────────────────────
if(CUSIFT_ENABLE_OPENMP)
  find_package(OpenMP COMPONENTS C CXX)
endif()

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

# Optimisation level is left to CMAKE_CUDA_FLAGS_<CONFIG>: nvcc forwards an
# explicit -O2 to the host compiler, and on MSVC that collides with the /RTC1
# in the Debug flags ("D8016: '/RTC1' and '/O2' are incompatible").
if(MSVC)
  target_compile_options(cusift_cuda PRIVATE
    --Werror cross-execution-space-call
    -Xcompiler=/W4 -Xcompiler=/Zc:preprocessor
  )
else()
  target_compile_options(cusift_cuda PRIVATE
    --Werror cross-execution-space-call
    -Xcompiler=-Wall,-Wextra
  )
endif()

if(CUSIFT_USE_FAST_MATH)
  target_compile_options(cusift_cuda PRIVATE --use_fast_math)
endif()

# ── cusift library (C/C++ sources + CUDA objects) ────────
if(CUSIFT_BUILD_SHARED)
  target_compile_definitions(cusift_cuda PRIVATE CUSIFT_EXPORTS)
  add_library(cusift SHARED ${C_SOURCES} $<TARGET_OBJECTS:cusift_cuda>)
  target_compile_definitions(cusift PRIVATE CUSIFT_EXPORTS)
  set_target_properties(cusift PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  cusift_self_contained(cusift)
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

# C/C++ compile options (no COMPILE_LANGUAGE guards needed — only C sources in
# this target).  No explicit /O2 or -O2: the per-configuration flags set it,
# and /O2 is incompatible with the /RTC1 that MSVC Debug builds carry.
if(MSVC)
  target_compile_options(cusift PRIVATE /W4 /Zc:preprocessor /D_CRT_SECURE_NO_WARNINGS)
  if(CUSIFT_ENABLE_AVX2 AND _CUSIFT_X86)
    set_source_files_properties(src/geomFuncs.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    message(STATUS "cusift: building geomFuncs.c with AVX2 + FMA (/arch:AVX2)")
  else()
    message(STATUS "cusift: building geomFuncs.c with scalar fallback (no AVX2)")
  endif()
else()
  target_compile_options(cusift PRIVATE -Wall -Wextra -Wpedantic)
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

# Link OpenMP if available.  The only OpenMP code is the host-side warp loop
# in cusift.cu, which nvcc compiles, so the flag has to be forwarded to the
# host compiler by hand: OpenMP::OpenMP_CXX only applies its flags to CXX
# sources and the cusift target has none.  This applies to MSVC too
# (-Xcompiler=-openmp), otherwise the pragma is silently ignored there and
# the CPU warp runs single-threaded.  Spaces become commas because nvcc
# splits -Xcompiler values on commas, not spaces.
if(OpenMP_CXX_FOUND)
  target_link_libraries(cusift PRIVATE OpenMP::OpenMP_CXX)
  if(OpenMP_CXX_FLAGS)
    string(REPLACE " " "," _cusift_omp_flags "${OpenMP_CXX_FLAGS}")
    target_compile_options(cusift_cuda PRIVATE "-Xcompiler=${_cusift_omp_flags}")
    unset(_cusift_omp_flags)
  endif()
endif()

# ── MATLAB façade: cusift_matlab shared library ─────────
# One self-contained shared library holding the whole of cusift (the same C
# sources and CUDA objects as the cusift target) plus the flat, handle-based
# surface of include/cusift_matlab.h that MATLAB's loadlibrary can parse.  It
# is deliberately not a MEX file: nothing in it is compiled with MATLAB's
# toolchain, and it is loaded with loadlibrary('cusift_matlab', ...).
if(CUSIFT_BUILD_MATLAB)
  add_library(cusift_matlab SHARED src/cusift_matlab.cpp ${C_SOURCES} $<TARGET_OBJECTS:cusift_cuda>)
  target_include_directories(cusift_matlab PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  )
  # The CUDA objects were compiled with CUSIFT_EXPORTS or CUSIFT_STATIC; the
  # façade's view of cusift.h has to match them.
  if(CUSIFT_BUILD_SHARED)
    target_compile_definitions(cusift_matlab PRIVATE CUSIFT_MATLAB_EXPORTS CUSIFT_EXPORTS)
  else()
    target_compile_definitions(cusift_matlab PRIVATE CUSIFT_MATLAB_EXPORTS CUSIFT_STATIC)
  endif()
  set_target_properties(cusift_matlab PROPERTIES
    OUTPUT_NAME cusift_matlab
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  if(MSVC)
    target_compile_options(cusift_matlab PRIVATE /W4 /Zc:preprocessor /D_CRT_SECURE_NO_WARNINGS)
  else()
    target_compile_options(cusift_matlab PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  target_link_libraries(cusift_matlab PRIVATE CUDA::cudart_static)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(cusift_matlab PRIVATE OpenMP::OpenMP_CXX)
  endif()
  cusift_self_contained(cusift_matlab)
  if(NOT CUSIFT_STATIC_RUNTIME)
    message(STATUS "cusift: cusift_matlab will import the C++ runtime (msvcp140.dll / libstdc++.so.6); set CUSIFT_STATIC_RUNTIME=ON for a library MATLAB can load regardless of its own runtime")
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

# Installed executables must find the installed shared library next door
# (<prefix>/bin -> <prefix>/lib) without LD_LIBRARY_PATH.  Ignored on Windows,
# where the DLL sits in the same bin/ directory.
if(CUSIFT_BUILD_SHARED)
  set_target_properties(cusift_demo cusift_speed PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
  )
endif()

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

if(CUSIFT_BUILD_MATLAB)
  # Not part of the cusiftTargets export: MATLAB loads it by file name.
  install(TARGETS cusift_matlab
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
  install(FILES include/cusift_matlab.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY matlab/ DESTINATION ${CMAKE_INSTALL_DATADIR}/cusift/matlab
    FILES_MATCHING PATTERN "*.m" PATTERN "*.md"
  )
endif()

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

# A static cusift carries its private link dependencies into the export as
# $<LINK_ONLY:...>, so the consumer's find_package(cusift) must locate the
# OpenMP::OpenMP_CXX target too; the config template reads this flag.
set(CUSIFT_CONFIG_NEEDS_OPENMP OFF)
if(NOT CUSIFT_BUILD_SHARED AND OpenMP_CXX_FOUND)
  set(CUSIFT_CONFIG_NEEDS_OPENMP ON)
endif()

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
