///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026 Jiarong Kang
//
// Developed at the Legged AI Lab, University of Wisconsin-Madison.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "crocoddyl/contact_id/config/contact-id-config.hpp"
#include "crocoddyl/contact_id/problem/contact-anitescu-id.hpp"

#include "contact_id_model.hpp"
#include "contact_id_outputs.hpp"
#include "contact_id_preprocess.hpp"

namespace {

struct ExtractedMarginal {
  crocoddyl::MarginalizedArrivalPrior prior;
  Eigen::MatrixXd raw_information;
  Eigen::VectorXd raw_gradient;
  Eigen::VectorXd theta_linearization;
  Eigen::VectorXd natural_vector;
  double min_eig;
  double max_eig;
  std::string status;
};

std::vector<double> make_solver_alphas(double alpha0) {
  std::vector<double> alphas;
  for (int i = 0; i < 11; ++i) {
    alphas.push_back(std::ldexp(alpha0, -i));
  }
  return alphas;
}

void configure_callbacks(crocoddyl::SolverFDDP& solver, bool verbose) {
  if (!verbose) {
    return;
  }
  std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract> > callbacks;
  callbacks.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
  solver.setCallbacks(callbacks);
}

std::string window_label(std::size_t window_index, std::size_t start_idx) {
  std::ostringstream os;
  os << "window_" << std::setw(3) << std::setfill('0') << window_index
     << "_start_" << std::setw(4) << std::setfill('0') << start_idx;
  return os.str();
}

crocoddyl::ContactIDOutputConfig window_outputs(
    const crocoddyl::ContactIDOutputConfig& base, std::size_t window_index,
    std::size_t start_idx) {
  crocoddyl::ContactIDOutputConfig out = base;
  out.directory = crocoddyl::contact_id_xml::join_path(
      base.directory, window_label(window_index, start_idx));
  return out;
}

Eigen::VectorXd flatten_parameter_logs(
    const std::vector<Eigen::VectorXd>& parameter_logs) {
  Eigen::VectorXd theta =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(10 * parameter_logs.size()));
  for (std::size_t i = 0; i < parameter_logs.size(); ++i) {
    if (parameter_logs[i].size() != 10) {
      throw std::runtime_error("Each parameter log block must have size 10.");
    }
    theta.segment(static_cast<Eigen::Index>(10 * i), 10) = parameter_logs[i];
  }
  return theta;
}

std::vector<Eigen::VectorXd> split_parameter_logs(const Eigen::VectorXd& theta) {
  if (theta.size() % 10 != 0) {
    throw std::runtime_error("Flattened parameter vector is not a multiple of 10.");
  }
  std::vector<Eigen::VectorXd> out(static_cast<std::size_t>(theta.size() / 10));
  for (std::size_t i = 0; i < out.size(); ++i) {
    out[i] = theta.segment(static_cast<Eigen::Index>(10 * i), 10);
  }
  return out;
}

void overwrite_theta(Eigen::VectorXd& x, Eigen::Index nq,
                     const Eigen::VectorXd& theta) {
  if (x.size() < nq + theta.size()) {
    throw std::runtime_error("State is too small for the parameter block.");
  }
  x.segment(nq, theta.size()) = theta;
}

void overwrite_theta_sequence(std::vector<Eigen::VectorXd>& xs, Eigen::Index nq,
                              const Eigen::VectorXd& theta) {
  for (std::size_t i = 0; i < xs.size(); ++i) {
    overwrite_theta(xs[i], nq, theta);
  }
}

void apply_prior_mean_to_data(const pinocchio::Model& model,
                              const Eigen::VectorXd& theta,
                              contact_id_experiment::PreparedContactIDData& data) {
  overwrite_theta(data.x0, model.nq, theta);
  overwrite_theta_sequence(data.state_task, model.nq, theta);
  overwrite_theta_sequence(data.state_init, model.nq, theta);
  data.parameter_logs = split_parameter_logs(theta);
}

Eigen::MatrixXd symmetrized(const Eigen::MatrixXd& matrix) {
  return 0.5 * (matrix + matrix.transpose());
}

Eigen::MatrixXd clamp_information(const Eigen::MatrixXd& information,
                                  double floor, double ceiling,
                                  double& min_eig, double& max_eig) {
  const Eigen::MatrixXd sym = symmetrized(information);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(sym);
  if (eig.info() != Eigen::Success) {
    throw std::runtime_error("Failed to decompose marginalized information.");
  }
  Eigen::VectorXd values = eig.eigenvalues();
  for (Eigen::Index i = 0; i < values.size(); ++i) {
    values[i] = std::min(ceiling, std::max(floor, values[i]));
  }
  min_eig = values.minCoeff();
  max_eig = values.maxCoeff();
  return eig.eigenvectors() * values.asDiagonal() * eig.eigenvectors().transpose();
}

double mass_from_theta(const Eigen::VectorXd& theta) {
  double mass = 0.;
  for (Eigen::Index i = 0; i < theta.size(); i += 10) {
    mass += contact_id_experiment::inertia_from_log_cholesky(
                theta.segment(i, 10))
                .mass();
  }
  return mass;
}

std::size_t shooting_knots(const crocoddyl::ContactIDSolverConfig& solver_cfg) {
  return (solver_cfg.horizon + solver_cfg.down_sample - 1) /
         solver_cfg.down_sample;
}

crocoddyl::ContactIDWeights without_arrival_regularization(
    const crocoddyl::ContactIDWeights& weights) {
  crocoddyl::ContactIDWeights out = weights;
  out.arrival_alpha = 0.;
  out.arrival_com = 0.;
  out.arrival_diag = 0.;
  return out;
}

crocoddyl::MarginalizedArrivalPrior initial_arrival_prior(
    const pinocchio::Model& model,
    const std::vector<pinocchio::JointIndex>& joints,
    const crocoddyl::ContactIDWeights& weights,
    const Eigen::VectorXd& theta) {
  const Eigen::Index np = theta.size();
  if (np != static_cast<Eigen::Index>(10 * joints.size())) {
    throw std::runtime_error("Initial theta size does not match identified joints.");
  }

  Eigen::MatrixXd information = Eigen::MatrixXd::Zero(np, np);
  const int stride = 10;
  for (std::size_t j = 0; j < joints.size(); ++j) {
    Eigen::VectorXd defect_weights = Eigen::VectorXd::Zero(stride);
    defect_weights[0] = std::pow(weights.arrival_alpha, 2);
    for (int i = 1; i <= 6; ++i) {
      defect_weights[i] = std::pow(weights.arrival_diag, 2);
    }
    for (int i = 7; i <= 9; ++i) {
      defect_weights[i] = std::pow(weights.arrival_com, 2);
    }
    const Eigen::VectorXd s_link =
        crocoddyl::computeFrozenSFromLogCholeskyJacobian(
            model, joints[j], weights.arrival_scale_alpha);
    if (s_link.size() != stride) {
      throw std::runtime_error("Initial prior scaling must have size 10.");
    }
    information
        .diagonal()
        .segment(static_cast<Eigen::Index>(stride * j), stride) =
        10. * defect_weights.array() * s_link.array().square();
  }
  return crocoddyl::MarginalizedArrivalPrior(theta, information);
}

crocoddyl::ContactIDOutputConfig quiet_outputs(
    const crocoddyl::ContactIDOutputConfig& outputs) {
  crocoddyl::ContactIDOutputConfig out = outputs;
  out.force_log.clear();
  out.log_initial_guess = false;
  out.rollout = false;
  return out;
}

void write_vector_csv(const std::string& path, const Eigen::VectorXd& vector) {
  std::ofstream os(path.c_str(), std::ios::out | std::ios::trunc);
  if (!os.is_open()) {
    throw std::runtime_error("Cannot open debug vector output: " + path);
  }
  os << std::setprecision(16);
  for (Eigen::Index i = 0; i < vector.size(); ++i) {
    if (i > 0) {
      os << ",";
    }
    os << vector[i];
  }
  os << "\n";
}

void write_matrix_csv(const std::string& path, const Eigen::MatrixXd& matrix) {
  std::ofstream os(path.c_str(), std::ios::out | std::ios::trunc);
  if (!os.is_open()) {
    throw std::runtime_error("Cannot open debug matrix output: " + path);
  }
  os << std::setprecision(16);
  for (Eigen::Index r = 0; r < matrix.rows(); ++r) {
    for (Eigen::Index c = 0; c < matrix.cols(); ++c) {
      if (c > 0) {
        os << ",";
      }
      os << matrix(r, c);
    }
    os << "\n";
  }
}

void save_prior_debug(
    const crocoddyl::ContactIDOutputConfig& outputs,
    const std::string& prefix,
    const crocoddyl::MarginalizedArrivalPrior& prior) {
  if (!prior.enabled) {
    return;
  }
  contact_id_experiment::ensure_dir(outputs.directory);
  write_vector_csv(crocoddyl::contact_id_xml::join_path(
                       outputs.directory, prefix + "_mean.csv"),
                   prior.mean);
  write_matrix_csv(crocoddyl::contact_id_xml::join_path(
                       outputs.directory, prefix + "_information.csv"),
                   prior.information);
  write_vector_csv(crocoddyl::contact_id_xml::join_path(
                       outputs.directory, prefix + "_eta.csv"),
                   prior.information * prior.mean);
}

void save_extraction_debug(
    const crocoddyl::ContactIDOutputConfig& outputs,
    const std::string& prefix,
    const ExtractedMarginal& marginal) {
  contact_id_experiment::ensure_dir(outputs.directory);
  save_prior_debug(outputs, prefix, marginal.prior);
  write_matrix_csv(crocoddyl::contact_id_xml::join_path(
                       outputs.directory, prefix + "_raw_information.csv"),
                   marginal.raw_information);
  write_vector_csv(crocoddyl::contact_id_xml::join_path(
                       outputs.directory, prefix + "_raw_gradient.csv"),
                   marginal.raw_gradient);
  write_vector_csv(crocoddyl::contact_id_xml::join_path(
                       outputs.directory, prefix + "_theta_linearization.csv"),
                   marginal.theta_linearization);
  write_vector_csv(crocoddyl::contact_id_xml::join_path(
                       outputs.directory, prefix + "_natural_vector.csv"),
                   marginal.natural_vector);
}

contact_id_experiment::PreparedContactIDData dropped_chunk_data(
    const contact_id_experiment::PreparedContactIDData& long_data,
    const std::vector<Eigen::VectorXd>& xs_solution,
    std::size_t dropped_knots) {
  if (dropped_knots == 0) {
    throw std::runtime_error("Dropped-horizon marginalization needs >0 knots.");
  }
  if (xs_solution.size() < dropped_knots + 1 ||
      long_data.state_task.size() < dropped_knots ||
      long_data.ctrl_task.size() + 1 < dropped_knots) {
    throw std::runtime_error(
        "Solved long-window trajectory is too short for the dropped horizon.");
  }

  contact_id_experiment::PreparedContactIDData out = long_data;
  out.x0 = xs_solution[0];
  out.state_task.assign(long_data.state_task.begin(),
                        long_data.state_task.begin() + dropped_knots);
  out.ctrl_task.assign(long_data.ctrl_task.begin(),
                       long_data.ctrl_task.begin() + dropped_knots - 1);
  out.state_init.assign(xs_solution.begin(),
                        xs_solution.begin() + dropped_knots + 1);
  out.ctrl_init.clear();
  return out;
}

boost::shared_ptr<crocoddyl::ShootingProblem> create_problem(
    const pinocchio::Model& model,
    const std::vector<pinocchio::JointIndex>& joints,
    const std::vector<std::string>& contact_frames,
    const crocoddyl::ContactIDWeights& weights,
    const crocoddyl::ContactIDSolverConfig& solver_cfg,
    const crocoddyl::ContactIDOutputConfig& outputs,
    const crocoddyl::MarginalizedArrivalPrior& prior,
    contact_id_experiment::PreparedContactIDData& data) {
  crocoddyl::ContactAnitescuIDProblem problem_builder(
      model, joints, contact_frames, weights,
      contact_id_experiment::force_log_path(outputs));
  if (prior.enabled) {
    problem_builder.set_arrival_prior(prior);
  }

  boost::shared_ptr<crocoddyl::ShootingProblem> shooting_problem =
      problem_builder.createEstimationProblem(
          data.x0, static_cast<double>(solver_cfg.down_sample) *
                       solver_cfg.interval,
          data.state_task.size(), data.state_task, data.ctrl_task);
  shooting_problem->set_nthreads(static_cast<int>(solver_cfg.n_thread));
  return shooting_problem;
}

ExtractedMarginal extract_dropped_chunk_information(
    const pinocchio::Model& model,
    const std::vector<pinocchio::JointIndex>& joints,
    const std::vector<std::string>& contact_frames,
    const crocoddyl::ContactIDXMLConfig& window_cfg,
    const contact_id_experiment::PreparedContactIDData& long_data,
    const std::vector<Eigen::VectorXd>& xs_solution,
    const std::vector<Eigen::VectorXd>& us_solution,
    std::size_t dropped_knots,
    double information_floor, double information_ceiling) {
  contact_id_experiment::PreparedContactIDData drop_data =
      dropped_chunk_data(long_data, xs_solution, dropped_knots);
  crocoddyl::ContactIDSolverConfig drop_solver = window_cfg.solver;
  drop_solver.horizon = dropped_knots * window_cfg.solver.down_sample;

  const crocoddyl::ContactIDWeights drop_weights =
      without_arrival_regularization(window_cfg.weights);
  const crocoddyl::ContactIDOutputConfig drop_outputs =
      quiet_outputs(window_cfg.outputs);

  boost::shared_ptr<crocoddyl::ShootingProblem> drop_problem =
      create_problem(model, joints, contact_frames, drop_weights, drop_solver,
                     drop_outputs, crocoddyl::MarginalizedArrivalPrior(),
                     drop_data);
  if (us_solution.size() < dropped_knots) {
    throw std::runtime_error(
        "Solved long-window controls are too short for dropped horizon.");
  }
  std::vector<Eigen::VectorXd> us_drop(us_solution.begin(),
                                       us_solution.begin() + dropped_knots);
  std::vector<Eigen::VectorXd> xs_feasible(drop_problem->get_T() + 1);
  drop_problem->rollout(us_drop, xs_feasible);

  Eigen::MatrixXd Lambda_raw;
  Eigen::VectorXd eta_raw;
  std::string status = "dropped_horizon_backward";
  try {
    crocoddyl::SolverFDDP extractor(drop_problem);
    extractor.set_alphas(make_solver_alphas(1.));
    extractor.setCandidate(xs_feasible, us_drop, true);
    extractor.calcDiff();
    extractor.backwardPass();
    Lambda_raw = extractor.get_Quu()[0];
    eta_raw = extractor.get_Qu()[0];
  } catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("Dropped-horizon backward extraction failed: ") + e.what());
  }

  double min_eig = 0.;
  double max_eig = 0.;
  const Eigen::MatrixXd information =
      clamp_information(Lambda_raw, information_floor, information_ceiling,
                        min_eig, max_eig);

  const Eigen::VectorXd theta_current =
      xs_feasible[1].segment(model.nq, us_drop[0].size());
  Eigen::LDLT<Eigen::MatrixXd> ldlt(information);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("Failed to solve dropped information system.");
  }
  const Eigen::VectorXd theta_mean = theta_current - ldlt.solve(eta_raw);

  ExtractedMarginal out;
  out.prior = crocoddyl::MarginalizedArrivalPrior(theta_mean, information);
  out.raw_information = Lambda_raw;
  out.raw_gradient = eta_raw;
  out.theta_linearization = theta_current;
  out.natural_vector = information * theta_mean;
  out.min_eig = min_eig;
  out.max_eig = max_eig;
  out.status = status;
  return out;
}

ExtractedMarginal accumulate_parameter_information(
    const crocoddyl::MarginalizedArrivalPrior& carried_prior,
    const ExtractedMarginal& dropped_information,
    double information_floor, double information_ceiling,
    double information_forgetting_factor) {
  const Eigen::Index np = dropped_information.prior.mean.size();
  Eigen::MatrixXd information =
      Eigen::MatrixXd::Zero(np, np);
  Eigen::VectorXd eta = Eigen::VectorXd::Zero(np);

  if (carried_prior.enabled) {
    if (carried_prior.mean.size() != np ||
        carried_prior.information.rows() != np ||
        carried_prior.information.cols() != np) {
      throw std::runtime_error(
          "Carried prior dimensions do not match dropped information.");
    }
    information.noalias() +=
        information_forgetting_factor * carried_prior.information;
    eta.noalias() +=
        information_forgetting_factor * carried_prior.information *
        carried_prior.mean;
  }

  information.noalias() += dropped_information.prior.information;
  eta.noalias() +=
      dropped_information.prior.information * dropped_information.prior.mean;

  double min_eig = 0.;
  double max_eig = 0.;
  const Eigen::MatrixXd clamped =
      clamp_information(information, information_floor, information_ceiling,
                        min_eig, max_eig);
  Eigen::LDLT<Eigen::MatrixXd> ldlt(clamped);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("Failed to solve accumulated information system.");
  }

  ExtractedMarginal out;
  out.prior = crocoddyl::MarginalizedArrivalPrior(ldlt.solve(eta), clamped);
  out.raw_information = information;
  out.raw_gradient = Eigen::VectorXd::Zero(np);
  out.theta_linearization = out.prior.mean;
  out.natural_vector = eta;
  out.min_eig = min_eig;
  out.max_eig = max_eig;
  out.status = dropped_information.status;
  return out;
}

void write_summary_header(std::ofstream& os, Eigen::Index np) {
  os << "window,start_idx,cost,feasibility,mass_estimate,min_info_eig,"
        "max_info_eig,extraction_status,information_forgetting_factor";
  for (Eigen::Index i = 0; i < np; ++i) {
    os << ",arrival_prior_diag_" << i;
  }
  for (Eigen::Index i = 0; i < np; ++i) {
    os << ",marginal_hessian_diag_" << i;
  }
  for (Eigen::Index i = 0; i < np; ++i) {
    os << ",theta_" << i;
  }
  os << "\n";
}

void write_summary_row(std::ofstream& os, std::size_t window_index,
                       std::size_t start_idx, double cost, double feasibility,
                       const crocoddyl::MarginalizedArrivalPrior& arrival_prior,
                       const ExtractedMarginal& marginal,
                       double information_forgetting_factor) {
  os << window_index << "," << start_idx << "," << cost << "," << feasibility
     << "," << mass_from_theta(marginal.prior.mean) << ","
     << marginal.min_eig << "," << marginal.max_eig << ","
     << marginal.status << "," << information_forgetting_factor;
  for (Eigen::Index i = 0; i < marginal.prior.mean.size(); ++i) {
    if (arrival_prior.enabled) {
      os << "," << arrival_prior.information(i, i);
    } else {
      os << ",nan";
    }
  }
  for (Eigen::Index i = 0; i < marginal.prior.mean.size(); ++i) {
    os << "," << marginal.prior.information(i, i);
  }
  for (Eigen::Index i = 0; i < marginal.prior.mean.size(); ++i) {
    os << "," << marginal.prior.mean[i];
  }
  os << "\n";
}

std::size_t available_windows(const crocoddyl::ContactIDXMLConfig& cfg) {
  const Eigen::Index q_rows =
      csvutil::readCSVtoEigen(cfg.data.q_csv).rows();
  const Eigen::Index v_rows =
      csvutil::readCSVtoEigen(cfg.data.v_csv).rows();
  const Eigen::Index u_rows =
      csvutil::readCSVtoEigen(cfg.data.u_csv).rows();
  const std::size_t rows = static_cast<std::size_t>(
      std::min(q_rows, std::min(v_rows, u_rows)));
  const std::size_t stride =
      cfg.moving_horizon.stride_knots * cfg.solver.down_sample;
  if (cfg.solver.start_idx + cfg.solver.horizon > rows) {
    return 0;
  }
  return 1 + (rows - cfg.solver.start_idx - cfg.solver.horizon) / stride;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: go2_sim_cz_m0p1_mhe <config.xml>\n";
      return 2;
    }

    const crocoddyl::ContactIDXMLConfig cfg =
        crocoddyl::contact_id_xml::load_config(argv[1]);
    contact_id_experiment::validate_required(cfg);

    const std::size_t max_available_windows = available_windows(cfg);
    if (max_available_windows == 0) {
      throw std::runtime_error("No valid moving-horizon windows fit the data.");
    }
    const std::size_t requested_windows =
        cfg.moving_horizon.enabled ? cfg.moving_horizon.max_windows : 1;
    const std::size_t n_windows =
        std::min(requested_windows, max_available_windows);
    const std::size_t n_shooting_knots = shooting_knots(cfg.solver);
    if (cfg.moving_horizon.stride_knots > n_shooting_knots) {
      throw std::runtime_error(
          "Moving-horizon stride_knots must be <= the down-sampled shooting "
          "horizon for dropped-chunk marginalization.");
    }

    if (cfg.solver.dry_run) {
      std::cout << "Parsed Go2 shifted-COM MHE contact-ID XML successfully.\n";
      std::cout << "contacts=" << cfg.contact_frames.size()
                << " identified_entries="
                << (cfg.identified_joint_indices.size() +
                    cfg.identified_joint_names.size())
                << "\n";
      std::cout << "moving_horizon_enabled=" << cfg.moving_horizon.enabled
                << " requested_windows=" << requested_windows
                << " available_windows=" << max_available_windows
                << " stride_knots=" << cfg.moving_horizon.stride_knots
                << " shooting_knots=" << n_shooting_knots
                << " information_forgetting_enabled="
                << cfg.moving_horizon.information_forgetting_enabled
                << " information_forgetting_factor="
                << cfg.moving_horizon.information_forgetting_factor
                << "\n";
      return 0;
    }

    std::cout
        << "Go2 shifted-COM MHE carries parameter information from the "
           "dropped horizon only. The long window remains the identification "
           "solve; the arrival prior is updated by a short dropped-chunk "
           "backward pass.\n";

    const pinocchio::Model model =
        contact_id_experiment::load_floating_base_model(cfg.robot);
    const std::vector<pinocchio::JointIndex> joints =
        contact_id_experiment::resolve_identified_joints(model, cfg);

    contact_id_experiment::clear_directory_contents(cfg.outputs.directory);
    std::cout << "Cleared MHE results directory: " << cfg.outputs.directory
              << "\n";
    const std::string summary_path = crocoddyl::contact_id_xml::join_path(
        cfg.outputs.directory, "mhe_summary.csv");
    std::ofstream summary(summary_path.c_str(), std::ios::out | std::ios::trunc);
    if (!summary.is_open()) {
      throw std::runtime_error("Cannot open MHE summary: " + summary_path);
    }
    summary << std::setprecision(12);
    write_summary_header(summary, static_cast<Eigen::Index>(10 * joints.size()));

    crocoddyl::MarginalizedArrivalPrior carried_prior;
    Eigen::VectorXd carried_theta;
    const double information_forgetting_factor =
        cfg.moving_horizon.information_forgetting_enabled
            ? cfg.moving_horizon.information_forgetting_factor
            : 1.;

    crocoddyl::Timer timer;
    for (std::size_t w = 0; w < n_windows; ++w) {
      crocoddyl::ContactIDXMLConfig window_cfg = cfg;
      window_cfg.solver.start_idx =
          cfg.solver.start_idx +
          w * cfg.moving_horizon.stride_knots * cfg.solver.down_sample;

      contact_id_experiment::PreparedContactIDData data =
          contact_id_experiment::prepare_contact_id_data(
              model, joints, cfg.contact_frames, window_cfg);
      if (w == 0) {
        carried_theta = flatten_parameter_logs(data.parameter_logs);
        carried_prior =
            initial_arrival_prior(model, joints, cfg.weights, carried_theta);
      } else {
        apply_prior_mean_to_data(model, carried_theta, data);
      }

      const crocoddyl::ContactIDOutputConfig outputs =
          window_outputs(cfg.outputs, w, window_cfg.solver.start_idx);
      contact_id_experiment::clear_outputs(outputs);
      contact_id_experiment::log_initial_guess(outputs, data.state_task,
                                               data.ctrl_task);
      save_prior_debug(outputs, "arrival_prior_used", carried_prior);

      std::cout << "Solving MHE window " << w + 1 << " / " << n_windows
                << " start_idx=" << window_cfg.solver.start_idx
                << " dense_prior=" << carried_prior.enabled
                << " information_forgetting_factor="
                << information_forgetting_factor << "\n";
      boost::shared_ptr<crocoddyl::ShootingProblem> shooting_problem =
          create_problem(model, joints, cfg.contact_frames, cfg.weights,
                         window_cfg.solver, outputs, carried_prior, data);

      crocoddyl::SolverFDDP solver(shooting_problem);
      solver.set_alphas(make_solver_alphas(cfg.solver.alpha0));
      configure_callbacks(solver, cfg.solver.callbacks);
      solver.solve(data.state_init, data.ctrl_init, cfg.solver.max_iter, false,
                   0.1);

      contact_id_experiment::save_solution_outputs(
          outputs, shooting_problem, solver.get_xs(), solver.get_us());
      contact_id_experiment::save_inertia_identification_report(
          outputs, model, joints, data.parameter_logs, solver.get_xs(),
          solver.get_us());

      const crocoddyl::MarginalizedArrivalPrior arrival_prior_used =
          carried_prior;
      const ExtractedMarginal dropped_information =
          extract_dropped_chunk_information(
              model, joints, cfg.contact_frames, window_cfg, data,
              solver.get_xs(), solver.get_us(),
              cfg.moving_horizon.stride_knots,
              cfg.moving_horizon.information_floor,
              cfg.moving_horizon.information_ceiling);
      save_extraction_debug(outputs, "dropped_information",
                            dropped_information);
      const ExtractedMarginal marginal =
          accumulate_parameter_information(
              carried_prior, dropped_information,
              cfg.moving_horizon.information_floor,
              cfg.moving_horizon.information_ceiling,
              information_forgetting_factor);
      save_extraction_debug(outputs, "accumulated_prior", marginal);
      carried_prior = marginal.prior;
      carried_theta = marginal.prior.mean;

      const double feasibility = solver.computeDynamicFeasibility();
      write_summary_row(summary, w, window_cfg.solver.start_idx,
                        solver.get_cost(), feasibility, arrival_prior_used,
                        marginal, information_forgetting_factor);
      summary.flush();
    }

    std::cout << "MHE windows solved: " << n_windows << "\n";
    std::cout << "MHE summary: " << summary_path << "\n";
    std::cout << "Duration: " << timer.get_duration() / 1000. << " seconds\n";
  } catch (const std::exception& e) {
    std::cerr << "go2_sim_cz_m0p1_mhe: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
