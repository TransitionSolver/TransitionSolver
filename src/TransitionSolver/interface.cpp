

#include <iostream>
#include "RSS.hpp"
#include "phasetracer.hpp"

int main() {

    LOGGER(error);

    EffectivePotential::RSS model(278002.0068332878, 40495.53405540935, 0.058281281662798376, 0.08498960515918183, -725.3805920400514, 0.4516261190913656, -114.20305903853114, 279137.2362500697, 40810.960318316844, 699.9301941045085, 425.62761968074693, 0.2987030761010101);
    model.set_daisy_method(EffectivePotential::DaisyMethod::Parwani);
    model.set_bUseBoltzmannSuppression(true);
    PhaseTracer::PhaseFinder pf(model);
    pf.set_t_high(1e4);
    pf.set_check_vacuum_at_high(false);
    pf.set_seed(0);
    pf.find_phases();
    PhaseTracer::TransitionFinder tf(pf);
    tf.find_transitions();
    std::cout << PhaseTracer::serialize(tf).str();
}
