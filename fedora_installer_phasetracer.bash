sudo dnf install alglib-devel nlopt-devel eigen3-devel boost-devel gsl-devel

PHASETRACER=${HOME}/.PhaseTracer

git -C ${PHASETRACER} pull || git clone --depth 1 https://github.com/PhaseTracer/PhaseTracer.git ${PHASETRACER}

mkdir -p $PHASETRACER/build
cd build
cmake -S ${PHASETRACER} -B ${PHASETRACER}/build
cmake --build ${PHASETRACER}/build
