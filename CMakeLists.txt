cmake_minimum_required(VERSION 3.18)

project(cudaSift
  VERSION 2.0.0
  LANGUAGES C CXX CUDA
)

# ── Options ──────────────────────────────────────────────
option(CUSIFT_BUILD_SHARED   "Build cusift as a shared library instead of static" OFF)

# ── CUDA architecture ───────────────────────────────────
# Override with -DCMAKE_CUDA_ARCHITECTURES="75;86" etc.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

# ── cusift library ───────────────────────────────────────
set(LIB_CUDA_SOURCES
  cudaImage.cu
  cudaSiftH.cu
  matching.cu
)

set(LIB_C_SOURCES
  geomFuncs.c
  cusfit.cu
)

if(CUSIFT_BUILD_SHARED)
  add_library(cusift SHARED ${LIB_CUDA_SOURCES} ${LIB_C_SOURCES})
  target_compile_definitions(cusift PRIVATE CUSIFT_EXPORTS)
  # Default-hidden visibility on GCC/Clang; only CUSIFT_API symbols are exported
  set_target_properties(cusift PROPERTIES
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
else()
  add_library(cusift STATIC ${LIB_CUDA_SOURCES} ${LIB_C_SOURCES})
  target_compile_definitions(cusift PUBLIC CUSIFT_STATIC)
endif()

# Public headers so consumers get the include path automatically
target_include_directories(cusift
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(cusift PRIVATE VERBOSE)

# Platform-specific flags
if(MSVC)
  target_compile_options(cusift PRIVATE
    $<$<COMPILE_LANGUAGE:C>:/W4 /O2 /D_CRT_SECURE_NO_WARNINGS>
    $<$<COMPILE_LANGUAGE:CXX>:/W4 /O2 /D_CRT_SECURE_NO_WARNINGS>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --Werror cross-execution-space-call -Xcompiler=/W4>
  )
  # AVX2 + FMA for geomFuncs.c
  set_source_files_properties(geomFuncs.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
else()
  target_compile_options(cusift PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-Wall -Wextra -Wpedantic -O2>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-O2 --Werror cross-execution-space-call -Xcompiler=-Wall,-Wextra>
  )
  # AVX2 + FMA for geomFuncs.c
  set_source_files_properties(geomFuncs.c PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

# cudart is linked automatically by CMake when CUDA is a project language

# ── Demo executable ──────────────────────────────────────
add_executable(cusift_demo test/main.cpp)
target_include_directories(cusift_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
target_link_libraries(cusift_demo PRIVATE cusift)

# ── Install ──────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS cusift
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(TARGETS cusift_demo
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
