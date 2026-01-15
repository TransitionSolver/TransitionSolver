#ifndef TRANSITION_SOLVER_RSS_BP_HPP_
#define TRANSITION_SOLVER_RSS_BP_HPP_

#include <vector>
#include "RSS.hpp"

namespace EffectivePotential {

class RSS_BP : public RSS {
 public:
    RSS_BP(std::vector<double> point) {
      // settings for benchmark points
      init();
      setParameters(point);
      set_daisy_method(EffectivePotential::DaisyMethod::Parwani);
      set_bUseBoltzmannSuppression(true);
      set_xi(0);
      set_useGSResummation(true);
    }
};

}  // end namespace EffectivePotential

#endif // TRANSITION_SOLVER_RSS_BP_HPP_
