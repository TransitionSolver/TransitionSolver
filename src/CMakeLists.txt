cmake_minimum_required(VERSION 4.2)
include(FetchContent)

set(HOME_DIR $ENV{HOME})
set(FETCHCONTENT_BASE_DIR "${HOME_DIR}/.TransitionSolver")
set(FETCHCONTENT_QUIET OFF)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  add_compile_definitions(
    EIGEN_DONT_VECTORIZE
    EIGEN_MAX_ALIGN_BYTES=16
    EIGEN_MAX_STATIC_ALIGN_BYTES=16
  )
endif()

FetchContent_Declare(
  PhaseTracer
  GIT_REPOSITORY https://github.com/PhaseTracer/PhaseTracer
  GIT_TAG fix_symmetric_partner
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(PhaseTracer)
