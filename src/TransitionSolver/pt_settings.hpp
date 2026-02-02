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

// Deep-merge: src overwrites destination (dst). Objects merge recursively. Arrays overwrite.
inline void merge_into(ptree& dst, const ptree& src) {
  // If src is an array or a scalar leaf, overwrite completely.
  if (is_array_node(src) || (src.empty() && !src.data().empty())) {
    dst = src;
    return;
  }

  // src is an object-like node.
  for (const auto& kv : src) {
    const std::string& key = kv.first;
    const ptree& src_child = kv.second;

    auto dst_child_opt = dst.get_child_optional(key);
    if (!dst_child_opt) {
      dst.put_child(key, src_child);
      continue;
    }

    ptree& dst_child = dst.get_child(key);

    // If either side is array, overwrite.
    if (is_array_node(src_child) || is_array_node(dst_child)) {
      dst.put_child(key, src_child);
      continue;
    }

    // If both are objects, recurse; otherwise overwrite.
    if (!src_child.empty() && !dst_child.empty()) {
      merge_into(dst_child, src_child);
    } else {
      dst.put_child(key, src_child);
    }
  }
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
  // Accept both:
  //  - "auto" / "rss_getMaxTemp" (scalar string)
  //  - {"mode":"rss_getMaxTemp", ...} (object)

  std::string mode;
  if (th.empty()) {
    mode = th.get_value<std::string>();
  } else {
    mode = th.get<std::string>("mode");
  }

  if (mode != "auto" && mode != "rss_getMaxTemp") {
    throw std::runtime_error("t_high auto mode must be 'auto' or 'rss_getMaxTemp', got: " + mode);
  }

  if constexpr (!has_getMaxTemp<ModelT>::value) {
    throw std::runtime_error(
      "phase_finder.t_high requested auto mode '" + mode +
      "', but this model does not implement getMaxTemp().");
  } else {
    // Defaults chosen to match old RSS behaviour
    const double margin      = th.empty() ? 1.01 : th.get<double>("margin", 1.01);
    const double hi_scale    = th.empty() ? 10.0 : th.get<double>("hi_scale", 10.0);
    const int n              = th.empty() ? 500  : th.get<int>("n", 500);
    const double dt_rel_s    = th.empty() ? 0.01 : th.get<double>("dt_rel_scale", 0.01);
    const double dt_abs_s    = th.empty() ? 0.001: th.get<double>("dt_abs_scale", 0.001);

    // Replicates what we had in RSS.cpp
    const double temperatureScale = model.get_temperature_scale();
    const double lo = temperatureScale;                 // matches old: temperatureScale
    const double hi = temperatureScale * hi_scale;      // matches old: temperatureScale * 10
    const double dt_rel = temperatureScale * dt_rel_s;  // matches old: temperatureScale * 0.01
    const double dt_abs = temperatureScale * dt_abs_s;  // matches old: temperatureScale * 0.001

    const double t_high = model.getMaxTemp(lo, hi, n, dt_rel, dt_abs) * margin;
    return t_high;
  }
}
  
// ---------- Strict apply of PhaseFinder + TransitionFinder settings ----------
// NOTE: These functions intentionally ERROR on unknown keys.

template <typename PhaseFinderT>
inline void apply_phase_finder_settings(PhaseFinderT& pf, const ptree& j) {
  for (const auto& kv : j) {
    const std::string& k = kv.first;
    const ptree& v = kv.second;

    // doubles
    if      (k == "x_abs_identical")             pf.set_x_abs_identical(v.get_value<double>());
    else if (k == "x_rel_identical")             pf.set_x_rel_identical(v.get_value<double>());
    else if (k == "x_abs_jump")                  pf.set_x_abs_jump(v.get_value<double>());
    else if (k == "x_rel_jump")                  pf.set_x_rel_jump(v.get_value<double>());
    else if (k == "find_min_x_tol_rel")          pf.set_find_min_x_tol_rel(v.get_value<double>());
    else if (k == "find_min_x_tol_abs")          pf.set_find_min_x_tol_abs(v.get_value<double>());
    else if (k == "find_min_max_f_eval")         pf.set_find_min_max_f_eval(v.get_value<double>());
    else if (k == "find_min_min_step")           pf.set_find_min_min_step(v.get_value<double>());
    else if (k == "find_min_max_time")           pf.set_find_min_max_time(v.get_value<double>());
    else if (k == "find_min_trace_abs_step")     pf.set_find_min_trace_abs_step(v.get_value<double>());
    else if (k == "find_min_locate_abs_step")    pf.set_find_min_locate_abs_step(v.get_value<double>());
    else if (k == "dt_start_rel")                pf.set_dt_start_rel(v.get_value<double>());
    else if (k == "dt_min_rel_split_phase")      pf.set_dt_min_rel_split_phase(v.get_value<double>());
    else if (k == "t_jump_rel")                  pf.set_t_jump_rel(v.get_value<double>());
    else if (k == "dt_max_abs")                  pf.set_dt_max_abs(v.get_value<double>());
    else if (k == "dt_max_rel")                  pf.set_dt_max_rel(v.get_value<double>());
    else if (k == "dt_min_rel")                  pf.set_dt_min_rel(v.get_value<double>());
    else if (k == "dt_min_abs")                  pf.set_dt_min_abs(v.get_value<double>());
    else if (k == "v")                           pf.set_v(v.get_value<double>());
    else if (k == "hessian_singular_rel_tol")     pf.set_hessian_singular_rel_tol(v.get_value<double>());
    else if (k == "linear_algebra_rel_tol")       pf.set_linear_algebra_rel_tol(v.get_value<double>());
    else if (k == "t_low")                        pf.set_t_low(v.get_value<double>());
    else if (k == "t_high")                       pf.set_t_high(v.get_value<double>());
    else if (k == "dt_merge_phases")             pf.set_dt_merge_phases(v.get_value<double>());
    else if (k == "dx_merge_phases")             pf.set_dx_merge_phases(v.get_value<double>());

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

    // vectors
    else if (k == "lower_bounds")                pf.set_lower_bounds(as_vector<double>(v));
    else if (k == "upper_bounds")                pf.set_upper_bounds(as_vector<double>(v));
    else {
      throw std::runtime_error("Unknown PhaseFinder setting: " + k);
    }
  }
}

template <typename TransitionFinderT>
inline void apply_transition_finder_settings(TransitionFinderT& tf, const ptree& j) {
  for (const auto& kv : j) {
    const std::string& k = kv.first;
    const ptree& v = kv.second;

    if      (k == "n_ew_scalars")                tf.set_n_ew_scalars(v.get_value<int>());
    else if (k == "separation")                  tf.set_separation(v.get_value<double>());
    else if (k == "assume_only_one_transition")  tf.set_assume_only_one_transition(v.get_value<bool>());
    else if (k == "TC_tol_rel")                  tf.set_TC_tol_rel(v.get_value<double>());
    else if (k == "max_iter")                    tf.set_max_iter(v.get_value<boost::uintmax_t>());
    else if (k == "change_rel_tol")              tf.set_change_rel_tol(v.get_value<double>());
    else if (k == "change_abs_tol")              tf.set_change_abs_tol(v.get_value<double>());
    else if (k == "calculate_action")            tf.set_calculate_action(v.get_value<bool>());

    // This one is used in your current interface.cpp already:
    else if (k == "check_subcritical_transitions") tf.set_check_subcritical_transitions(v.get_value<bool>());

    else {
      throw std::runtime_error("Unknown TransitionFinder setting: " + k);
    }
  }
}

} // namespace TS_PT
