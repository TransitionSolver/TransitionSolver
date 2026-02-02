// template for calling models

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <eigen3/Eigen/Eigenvalues>
#include "phasetracer.hpp"

#include "pt_settings.hpp"

#include MODEL_HEADER

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


// Strict top-level section validation
static void validate_top_level_sections_strict(const TS_PT::ptree& cfg) {
  for (const auto& kv : cfg) {
    const std::string& k = kv.first;
    if (k != "phase_finder" && k != "transition_finder") {
      throw std::runtime_error("Unknown top-level PT settings section: " + k);
    }
  }
}


// Parse/compute t_high from settings if present.
// - if no config or no phase_finder.t_high: returns fallback_t_high
// - if numeric: uses numeric
// - if string "auto" / "rss_getMaxTemp": uses TS_PT::compute_t_high_auto(model, node)
// - if object {"mode": "...", ...}: uses TS_PT::compute_t_high_auto(model, node)
template <typename ModelT>
static double resolve_t_high_from_config(
    ModelT& model,
    TS_PT::ptree& phase_finder_node,   // passed by ref so we can erase("t_high")
    double fallback_t_high) {

  auto th_opt = phase_finder_node.get_child_optional("t_high");
  if (!th_opt) {
    return fallback_t_high;
  }

  const TS_PT::ptree& th = *th_opt;

  double t_high_effective = fallback_t_high;

  // property_tree represents numbers as leaf strings; objects have children.
  if (th.empty()) {
    // leaf: could be "1200" or "auto"
    const std::string raw = th.get_value<std::string>();

    // try numeric
    try {
      size_t idx = 0;
      double val = std::stod(raw, &idx);
      if (idx == raw.size()) {
        t_high_effective = val;
      } else {
        // trailing junk => treat as mode string
        t_high_effective = TS_PT::compute_t_high_auto(model, th);
      }
    } catch (...) {
      // not numeric => mode string ("auto"/"rss_getMaxTemp")
      t_high_effective = TS_PT::compute_t_high_auto(model, th);
    }
  } else {
    // object form => must contain "mode"
    t_high_effective = TS_PT::compute_t_high_auto(model, th);
  }

  // Remove t_high so strict apply_phase_finder_settings doesn't see a non-double
  phase_finder_node.erase("t_high");
  return t_high_effective;
}


int main(int argc, char* argv[]) {
  LOGGER(error);

  try {
    // Backwards compatible:
    //   <exe> <point_file> <t_high>
    //   <exe> <point_file> <t_high> <pt_settings.json>
    if (!(argc == 3 || argc == 4)) {
      throw std::runtime_error("Usage: <exe> <point_file> <t_high> [pt_settings.json]");
    }

    const std::string point_file = argv[1];
    const double t_high_arg = std::atof(argv[2]);
    const bool has_settings = (argc == 4);
    const std::string settings_file = has_settings ? std::string(argv[3]) : std::string();

    const std::vector<double> point = read_point(point_file);

    // Construct the model from the numeric point vector.
    // (This is what your generated interface expects for RSS_BP-style wrappers and your new ToyModel vector ctor.)
    auto model = MODEL_NAME_WITH_NAMESPACE(point);

    // --- Load settings if provided (single merged JSON from Python layer) ---
    TS_PT::ptree cfg;
    TS_PT::ptree phase_finder_cfg;
    TS_PT::ptree transition_finder_cfg;

    if (has_settings) {
      cfg = TS_PT::read_json_file(settings_file);
      validate_top_level_sections_strict(cfg);

      if (auto pf_node = cfg.get_child_optional("phase_finder")) {
        phase_finder_cfg = *pf_node;
      }
      if (auto tf_node = cfg.get_child_optional("transition_finder")) {
        transition_finder_cfg = *tf_node;
      }
    }

    // --- PhaseFinder ---
    PhaseTracer::PhaseFinder pf(model);

    // TS-side defaults (users can override via JSON)
    pf.set_check_vacuum_at_high(false);
    pf.set_seed(0);
    pf.set_check_hessian_singular(false);
    pf.set_check_merge_phase_gaps(false);

    // Determine effective t_high:
    //  - start from CLI t_high
    //  - allow JSON override or RSS-auto mode
    double t_high_effective = t_high_arg;
    if (has_settings && !phase_finder_cfg.empty()) {
      t_high_effective = resolve_t_high_from_config(model, phase_finder_cfg, t_high_arg);
    }
    pf.set_t_high(t_high_effective);

    // Apply remaining PhaseFinder settings strictly (unknown keys -> error)
    if (has_settings && !phase_finder_cfg.empty()) {
      TS_PT::apply_phase_finder_settings(pf, phase_finder_cfg);
    }

    pf.find_phases();

    // --- TransitionFinder ---
    PhaseTracer::TransitionFinder tf(pf);

    // TS-side defaults (users can override via JSON)
    tf.set_check_subcritical_transitions(false);
    tf.set_assume_only_one_transition(true);

    // Apply TransitionFinder settings strictly (unknown keys -> error)
    if (has_settings && !transition_finder_cfg.empty()) {
      TS_PT::apply_transition_finder_settings(tf, transition_finder_cfg);
    }

    tf.find_transitions();
    tf.find_transition_paths();

    std::cout << PhaseTracer::serialize(tf);
    return 0;

  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
