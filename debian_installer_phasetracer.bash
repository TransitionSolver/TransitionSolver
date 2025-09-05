sudo apt install git cmake libalglib-dev libnlopt-cxx-dev libeigen3-dev libboost-filesystem-dev libboost-log-dev libgsl-dev

PHASETRACER=${HOME}/.PhaseTracer

git -C ${PHASETRACER} pull || git clone --depth 1 https://github.com/PhaseTracer/PhaseTracer.git ${PHASETRACER}

mkdir -p $PHASETRACER/build
cd build
cmake -S ${PHASETRACER} -B ${PHASETRACER}/build
cmake --build ${PHASETRACER}/build
