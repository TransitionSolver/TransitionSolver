cmake_minimum_required(VERSION 4.2)
include(FetchContent)

set(HOME_DIR $ENV{HOME})
set(FETCHCONTENT_BASE_DIR "${HOME_DIR}/.TransitionSolver")
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  PhaseTracer
  GIT_REPOSITORY https://github.com/PhaseTracer/PhaseTracer
  GIT_TAG c4b5960ced478cb1079707f380f7d19003178a1c
  GIT_SHALLOW FALSE
  GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(PhaseTracer)
