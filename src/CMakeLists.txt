cmake_minimum_required(VERSION 4.2)
include(FetchContent)

set(HOME_DIR $ENV{HOME})
set(FETCHCONTENT_BASE_DIR "${HOME_DIR}/.TransitionSolver")
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  PhaseTracer
  GIT_REPOSITORY https://github.com/PhaseTracer/PhaseTracer
  GIT_TAG master
  GIT_SHALLOW TRUE
  GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(PhaseTracer)
