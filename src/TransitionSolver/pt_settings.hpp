#pragma once

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <nlopt.hpp>

#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <type_traits>

namespace TS_PT {

using boost::property_tree::ptree;

// ---------- JSON helpers ----------
inline ptree read_json_file(const std::string& path) {
  ptree root;
  boost::property_tree::read_json(path, root);
  return root;
}

inline bool is_array_node(const ptree& node) {
  if (node.empty()) return false;
  for (const auto& kv : node) {
    // arrays use empty keys, if non-empty its an object or dictionary
    if (!kv.first.empty()) return false; 
  }
  return true;
}


// Parse JSON array -> std::vector<T>
template <typename T>
inline std::vector<T> as_vector(const ptree& node) {
  if (!is_array_node(node)) {
    throw std::runtime_error("Expected JSON array.");
  }
  std::vector<T> out;
  out.reserve(node.size());
  for (const auto& kv : node) {
    out.push_back(kv.second.get_value<T>());
  }
  return out;
}

// check we have "scale_with" key
inline bool is_scale_rule(const ptree& node) {
  if (node.empty()) return false;

  bool has_scale_with = false;
  for (const auto& kv : node) {
    const std::string& key = kv.first;

    if (key == "scale_with") {
      has_scale_with = true; // this key must be present 
    } else if (key != "factor" && key != "offset") {
      return false; //return false immediately if invalid key
    }

    // Require leaf values for all entries in the rule object
    if (!kv.second.empty()) return false;
  }

  return has_scale_with;
}

// get temperature_scale or field_scale from model if requested
// needed for scaling settings that are sensitive to the scale  
template <typename ModelT>
double get_named_scale(const ModelT& model, const std::string& name) {
  if (name == "field_scale") {
    return model.get_field_scale();
  }
  if (name == "temperature_scale") {
    return model.get_temperature_scale();
  }
  throw std::runtime_error("Unknown scale_with target: " + name);
}

// routine for resolving whether the setting should be applied as a number
// Or if the user settings provide a scale factor based on temperature_scale or field_scale  
template <typename ModelT>
double resolve_double(const ptree& node, const ModelT& model, const std::string& context) {
  // Plain numeric leaf
  if (node.empty()) {
    try {
      return node.get_value<double>();
    } catch (const std::exception&) {
      throw std::runtime_error("Expected numeric value for " + context);
    }
  }

  // Generic scaling rule
  if (is_scale_rule(node)) {
    const std::string scale_with = node.get<std::string>("scale_with");
    const double factor = node.get<double>("factor", 1.0);
    const double offset = node.get<double>("offset", 0.0);
    return factor * get_named_scale(model, scale_with) + offset;
  }

  throw std::runtime_error("Expected numeric value or scale rule for " + context);
}

// As above but for vector doubles.
template <typename ModelT>
std::vector<double> resolve_vector_double(const ptree& node, const ModelT& model,
                                          const std::string& context) {
  if (!is_array_node(node)) {
    throw std::runtime_error("Expected JSON array for " + context);
  }

  std::vector<double> out;
  out.reserve(node.size());
  for (const auto& kv : node) {
    out.push_back(resolve_double(kv.second, model, context + "[]"));
  }
  return out;
}
  

// nlopt::algorithm parsing: allow "LN_SBPLX" or "nlopt::LN_SBPLX" or integer.
inline nlopt::algorithm parse_nlopt_algorithm(std::string s) {
  // trim spaces
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();

  // numeric?
  bool all_digits = !s.empty();
  for (char c : s) {
    if (!(std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+')) { all_digits = false; break; }
  }
  if (all_digits) {
    int v = std::stoi(s);
    return static_cast<nlopt::algorithm>(v);
  }

  // strip "nlopt::"
  const std::string prefix = "nlopt::";
  if (s.rfind(prefix, 0) == 0) s = s.substr(prefix.size());

  static const std::unordered_map<std::string, nlopt::algorithm> map = {
    {"LN_SBPLX", nlopt::LN_SBPLX},
    {"LN_NELDERMEAD", nlopt::LN_NELDERMEAD},
    {"LN_PRAXIS", nlopt::LN_PRAXIS},
    {"LN_BOBYQA", nlopt::LN_BOBYQA},
    {"LN_COBYLA", nlopt::LN_COBYLA},
    {"LD_MMA", nlopt::LD_MMA},
    {"LD_SLSQP", nlopt::LD_SLSQP},
    {"LD_LBFGS", nlopt::LD_LBFGS},
    {"GN_DIRECT", nlopt::GN_DIRECT},
    {"GN_DIRECT_L", nlopt::GN_DIRECT_L},
    {"GN_CRS2_LM", nlopt::GN_CRS2_LM},
  };

  auto it = map.find(s);
  if (it == map.end()) {
    throw std::runtime_error("Unknown nlopt::algorithm: " + s);
  }
  return it->second;
}

// Check for existence of getMaxTemp method
// This is necessary because we currently need RSS specific methiod getMaxTemp
// for RSS model  making it very hard to write general interface
// ToDo: find a different way to set t_high for RSS and remove these complications  
template <typename T, typename = void>
struct has_getMaxTemp : std::false_type {};

template <typename T>
struct has_getMaxTemp<T, std::void_t<
  decltype(std::declval<T>().getMaxTemp(
    std::declval<double>(),
    std::declval<double>(),
    std::declval<int>(),
    std::declval<double>(),
    std::declval<double>()
  ))
>> : std::true_type {};

  // compute the t_high checking for rss specific appraoch
  // ToDo: find a different way to set t_high for RSS and remove these complications
template <typename ModelT>
double compute_t_high_auto(ModelT& model, const boost::property_tree::ptree& th) {
  if (!th.empty()) {
    throw std::runtime_error(
      "phase_finder.t_high auto mode must be a string: 'auto' or 'rss_getMaxTemp'.");
  }

  const std::string mode = th.get_value<std::string>();
  if (mode != "auto" && mode != "rss_getMaxTemp") {
    throw std::runtime_error("t_high auto mode must be 'auto' or 'rss_getMaxTemp', got: " + mode);
  }

  if constexpr (!has_getMaxTemp<ModelT>::value) {
    throw std::runtime_error(
      "phase_finder.t_high requested auto mode '" + mode +
      "', but this model does not implement getMaxTemp().");
  } else {
    const double temperatureScale = model.get_temperature_scale();
    const double lo = temperatureScale;
    const double hi = temperatureScale * 10.0;
    const int n = 500;
    const double precision = temperatureScale * 0.01;
    const double finiteDifferenceStepSize = temperatureScale * 0.001;
    return model.getMaxTemp(lo, hi, n, precision, finiteDifferenceStepSize) * 1.01;
  }
}  
  
// ---------- Strict apply of PhaseFinder + TransitionFinder settings ----------
// NOTE: These functions intentionally ERROR on unknown keys.
template <typename PhaseFinderT, typename ModelT>
inline void apply_phase_finder_settings(PhaseFinderT& pf, const ptree& j, const ModelT& model) {
  for (const auto& kv : j) {
    const std::string& k = kv.first;
    const ptree& v = kv.second;

    // doubles
    if      (k == "x_abs_identical")              pf.set_x_abs_identical(resolve_double(v, model, "phase_finder.x_abs_identical"));
    else if (k == "x_rel_identical")              pf.set_x_rel_identical(resolve_double(v, model, "phase_finder.x_rel_identical"));
    else if (k == "x_abs_jump")                   pf.set_x_abs_jump(resolve_double(v, model, "phase_finder.x_abs_jump"));
    else if (k == "x_rel_jump")                   pf.set_x_rel_jump(resolve_double(v, model, "phase_finder.x_rel_jump"));
    else if (k == "find_min_x_tol_rel")           pf.set_find_min_x_tol_rel(resolve_double(v, model, "phase_finder.find_min_x_tol_rel"));
    else if (k == "find_min_x_tol_abs")           pf.set_find_min_x_tol_abs(resolve_double(v, model, "phase_finder.find_min_x_tol_abs"));
    else if (k == "find_min_max_f_eval")          pf.set_find_min_max_f_eval(resolve_double(v, model, "phase_finder.find_min_max_f_eval"));
    else if (k == "find_min_min_step")            pf.set_find_min_min_step(resolve_double(v, model, "phase_finder.find_min_min_step"));
    else if (k == "find_min_max_time")            pf.set_find_min_max_time(resolve_double(v, model, "phase_finder.find_min_max_time"));
    else if (k == "find_min_trace_abs_step")      pf.set_find_min_trace_abs_step(resolve_double(v, model, "phase_finder.find_min_trace_abs_step"));
    else if (k == "find_min_locate_abs_step")     pf.set_find_min_locate_abs_step(resolve_double(v, model, "phase_finder.find_min_locate_abs_step"));
    else if (k == "dt_start_rel")                 pf.set_dt_start_rel(resolve_double(v, model, "phase_finder.dt_start_rel"));
    else if (k == "dt_min_rel_split_phase")       pf.set_dt_min_rel_split_phase(resolve_double(v, model, "phase_finder.dt_min_rel_split_phase"));
    else if (k == "t_jump_rel")                   pf.set_t_jump_rel(resolve_double(v, model, "phase_finder.t_jump_rel"));
    else if (k == "dt_max_abs")                   pf.set_dt_max_abs(resolve_double(v, model, "phase_finder.dt_max_abs"));
    else if (k == "dt_max_rel")                   pf.set_dt_max_rel(resolve_double(v, model, "phase_finder.dt_max_rel"));
    else if (k == "dt_min_rel")                   pf.set_dt_min_rel(resolve_double(v, model, "phase_finder.dt_min_rel"));
    else if (k == "dt_min_abs")                   pf.set_dt_min_abs(resolve_double(v, model, "phase_finder.dt_min_abs"));
    else if (k == "v")                            pf.set_v(resolve_double(v, model, "phase_finder.v"));
    else if (k == "hessian_singular_rel_tol")     pf.set_hessian_singular_rel_tol(resolve_double(v, model, "phase_finder.hessian_singular_rel_tol"));
    else if (k == "linear_algebra_rel_tol")       pf.set_linear_algebra_rel_tol(resolve_double(v, model, "phase_finder.linear_algebra_rel_tol"));
    else if (k == "t_low")                        pf.set_t_low(resolve_double(v, model, "phase_finder.t_low"));
    else if (k == "t_high")                       pf.set_t_high(resolve_double(v, model, "phase_finder.t_high"));
    else if (k == "dt_merge_phases")              pf.set_dt_merge_phases(resolve_double(v, model, "phase_finder.dt_merge_phases"));
    else if (k == "dx_merge_phases")              pf.set_dx_merge_phases(resolve_double(v, model, "phase_finder.dx_merge_phases"));

    // vectors
    else if (k == "lower_bounds")                 pf.set_lower_bounds(resolve_vector_double(v, model, "phase_finder.lower_bounds"));
    else if (k == "upper_bounds")                 pf.set_upper_bounds(resolve_vector_double(v, model, "phase_finder.upper_bounds"));

    // nlopt algorithm
    else if (k == "find_min_algorithm")          pf.set_find_min_algorithm(parse_nlopt_algorithm(v.get_value<std::string>()));

    // sizes / ints
    else if (k == "n_test_points")               pf.set_n_test_points(v.get_value<std::size_t>());
    else if (k == "n_ew_scalars")                pf.set_n_ew_scalars(v.get_value<std::size_t>());
    else if (k == "seed")                        pf.set_seed(v.get_value<int>());
    else if (k == "trace_max_iter")              pf.set_trace_max_iter(v.get_value<unsigned int>());

    // bools
    else if (k == "check_vacuum_at_low")         pf.set_check_vacuum_at_low(v.get_value<bool>());
    else if (k == "check_vacuum_at_high")        pf.set_check_vacuum_at_high(v.get_value<bool>());
    else if (k == "check_dx_min_dt")             pf.set_check_dx_min_dt(v.get_value<bool>());
    else if (k == "check_hessian_singular")      pf.set_check_hessian_singular(v.get_value<bool>());
    else if (k == "check_merge_phase_gaps")      pf.set_check_merge_phase_gaps(v.get_value<bool>());

    else {
      throw std::runtime_error("Unknown PhaseFinder setting: " + k);
    }
  }
}

template <typename TransitionFinderT, typename ModelT>
inline void apply_transition_finder_settings(TransitionFinderT& tf, const ptree& j, const ModelT& model) {
  for (const auto& kv : j) {
    const std::string& k = kv.first;
    const ptree& v = kv.second;

    if      (k == "n_ew_scalars")                 tf.set_n_ew_scalars(v.get_value<int>());
    else if (k == "separation")                   tf.set_separation(resolve_double(v, model, "transition_finder.separation"));
    else if (k == "assume_only_one_transition")   tf.set_assume_only_one_transition(v.get_value<bool>());
    else if (k == "TC_tol_rel")                   tf.set_TC_tol_rel(resolve_double(v, model, "transition_finder.TC_tol_rel"));
    else if (k == "max_iter")                     tf.set_max_iter(v.get_value<boost::uintmax_t>());
    else if (k == "change_rel_tol")               tf.set_change_rel_tol(resolve_double(v, model, "transition_finder.change_rel_tol"));
    else if (k == "change_abs_tol")               tf.set_change_abs_tol(resolve_double(v, model, "transition_finder.change_abs_tol"));
    else if (k == "check_subcritical_transitions") tf.set_check_subcritical_transitions(v.get_value<bool>());

    else {
      throw std::runtime_error("Unknown TransitionFinder setting: " + k);
    }
  }
}

} // namespace TS_PT
