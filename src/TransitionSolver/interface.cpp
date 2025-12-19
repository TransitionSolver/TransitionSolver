// template for calling models

#include <iostream>
#include <stdexcept>
#include <vector>

#include "phasetracer.hpp"


std::vector<double> read_point(std::string file_name) {
    std::ifstream file(file_name);
    if (!file) {
        throw std::runtime_error("Could not open file");
    }

    std::vector<double> point;
    double num;
    while (file >> num) {
        point.push_back(num);
    }
    return point;
}


int main(int argc, char *argv[]) {

    LOGGER(error);
    
    if (argc == 1) {
      throw std::runtime_error("Need file name of parameter point");
    }
    
    std::vector<double> point = read_point(argv[1]);
    MODEL_NAME_WITH_NAMEPSPACE model(point);
    model.set_daisy_method(EffectivePotential::DaisyMethod::Parwani);
    model.set_bUseBoltzmannSuppression(false);
    model.set_xi(0);
    model.set_useGSResummation(true);
    
    PhaseTracer::PhaseFinder pf(model);
    
    pf.set_t_high(253.20465304119307);
    pf.set_check_vacuum_at_high(false);
    pf.set_seed(0);
    pf.set_check_hessian_singular(false);
    pf.set_check_merge_phase_gaps(false);
    
    pf.find_phases();
    
    PhaseTracer::TransitionFinder tf(pf);
    tf.set_check_subcritical_transitions(false);
    tf.set_assume_only_one_transition(true);
    
    tf.find_transitions();
    
    tf.find_transition_paths(model, true);
    
    std::cout << PhaseTracer::serialize(tf).str();
}
