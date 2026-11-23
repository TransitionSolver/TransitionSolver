project("ts-pt-installer")
cmake_minimum_required(VERSION 4.2)
include(FetchContent)

set(FETCHCONTENT_QUIET OFF)


set(FETCHCONTENT_BASE_DIR "$ENV{TS_INSTALL_DIR}")
if(NOT FETCHCONTENT_BASE_DIR )
  set(FETCHCONTENT_BASE_DIR "$ENV{HOME}/.TransitionSolver")
endif()

message(STATUS "TS install directory: ${FETCHCONTENT_BASE_DIR}")

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
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)

set(OVERWRITE "$ENV{TS_INSTALL_OVERWRITE}")

if(EXISTS "${FETCHCONTENT_BASE_DIR}/phasetracer-src" AND NOT OVERWRITE)
  message(WARNING "PhaseTracer already seems to be present at ${FETCHCONTENT_BASE_DIR}/phasetracer-src. The installer will build the existing source files without fetching updates. To use the latest PhaseTracer, update the source in that directory and rebuild.")
  add_subdirectory(
    "${FETCHCONTENT_BASE_DIR}/phasetracer-src/"
    "${FETCHCONTENT_BASE_DIR}/phasetracer-build/"
  )
else()
  FetchContent_MakeAvailable(PhaseTracer)
endif()
