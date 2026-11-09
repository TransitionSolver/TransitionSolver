project("ts-pt-installer")
cmake_minimum_required(VERSION 4.2)
include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

set(FETCHCONTENT_BASE_DIR "$ENV{TS_INSTALL_DIR}")
if(NOT FETCHCONTENT_BASE_DIR )
  set(FETCHCONTENT_BASE_DIR "$ENV{HOME}/.TransitionSolver")
endif()

message(STATUS "TS install directory: ${FETCHCONTENT_BASE_DIR}")

FetchContent_Declare(
  PhaseTracer
  GIT_REPOSITORY https://github.com/PhaseTracer/PhaseTracer
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)

set(OVERWRITE "$ENV{TS_INSTALL_OVERWRITE}")

if(EXISTS "${FETCHCONTENT_BASE_DIR}/phasetracer-src" AND NOT OVERWRITE)
  message(WARNING "PhaseTracer already seems to be present. Not modifying existing source files in '${FETCHCONTENT_BASE_DIR}/phasetracer-src/'")
  add_subdirectory(
    "${FETCHCONTENT_BASE_DIR}/phasetracer-src/"
    "${FETCHCONTENT_BASE_DIR}/phasetracer-build/"
  )
else()
  FetchContent_MakeAvailable(PhaseTracer)
endif()
