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

ExtractedMarginal extract_marginal(
    const boost::shared_ptr<crocoddyl::ShootingProblem>& shooting_problem,
    const crocoddyl::SolverFDDP& solved_solver,
    const std::vector<Eigen::VectorXd>& xs_solution,
    const std::vector<Eigen::VectorXd>& us_solution,
    Eigen::Index nq,
    double information_floor, double information_ceiling) {
  if (xs_solution.size() < 2 || us_solution.empty()) {
    throw std::runtime_error("Cannot extract marginal from an empty solution.");
  }

  std::vector<Eigen::VectorXd> xs_feasible(shooting_problem->get_T() + 1);
  shooting_problem->rollout(us_solution, xs_feasible);

  Eigen::MatrixXd Lambda_raw;
  Eigen::VectorXd eta_raw;
  std::string status = "fresh_backward";
  try {
    crocoddyl::SolverFDDP extractor(shooting_problem);
    extractor.set_alphas(make_solver_alphas(1.));
    extractor.setCandidate(xs_feasible, us_solution, true);
    extractor.calcDiff();
    extractor.backwardPass();
    Lambda_raw = extractor.get_Quu()[0];
    eta_raw = extractor.get_Qu()[0];
  } catch (const std::exception&) {
    status = "cached_solver_backward";
    Lambda_raw = solved_solver.get_Quu()[0];
    eta_raw = solved_solver.get_Qu()[0];
  }

  double min_eig = 0.;
  double max_eig = 0.;
  const Eigen::MatrixXd information =
      clamp_information(Lambda_raw, information_floor, information_ceiling,
                        min_eig, max_eig);

  const Eigen::VectorXd theta_current =
      xs_feasible[1].segment(nq, us_solution[0].size());
  Eigen::LDLT<Eigen::MatrixXd> ldlt(information);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("Failed to solve marginalized information system.");
  }
  const Eigen::VectorXd theta_mean = theta_current - ldlt.solve(eta_raw);

  ExtractedMarginal out;
  out.prior = crocoddyl::MarginalizedArrivalPrior(theta_mean, information);
  out.min_eig = min_eig;
  out.max_eig = max_eig;
  out.status = status;
  return out;
}

void write_summary_header(std::ofstream& os, Eigen::Index np) {
  os << "window,start_idx,cost,feasibility,mass_estimate,min_info_eig,"
        "max_info_eig,extraction_status";
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
                       const ExtractedMarginal& marginal) {
  os << window_index << "," << start_idx << "," << cost << "," << feasibility
     << "," << mass_from_theta(marginal.prior.mean) << ","
     << marginal.min_eig << "," << marginal.max_eig << ","
     << marginal.status;
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
                << "\n";
      return 0;
    }

    std::cout
        << "Go2 shifted-COM MHE uses one-knot overlapping windows by default. Carrying "
           "full Quu[0] is a filtering-style marginalization experiment, not "
           "an independent non-overlap batch posterior.\n";

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
      } else {
        apply_prior_mean_to_data(model, carried_theta, data);
      }

      const crocoddyl::ContactIDOutputConfig outputs =
          window_outputs(cfg.outputs, w, window_cfg.solver.start_idx);
      contact_id_experiment::clear_outputs(outputs);
      contact_id_experiment::log_initial_guess(outputs, data.state_task,
                                               data.ctrl_task);

      std::cout << "Solving MHE window " << w + 1 << " / " << n_windows
                << " start_idx=" << window_cfg.solver.start_idx
                << " dense_prior=" << carried_prior.enabled << "\n";
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
      ExtractedMarginal marginal =
          extract_marginal(shooting_problem, solver, solver.get_xs(),
                           solver.get_us(), model.nq,
                           cfg.moving_horizon.information_floor,
                           cfg.moving_horizon.information_ceiling);
      carried_prior = marginal.prior;
      carried_theta = marginal.prior.mean;

      const double feasibility = solver.computeDynamicFeasibility();
      write_summary_row(summary, w, window_cfg.solver.start_idx,
                        solver.get_cost(), feasibility, arrival_prior_used,
                        marginal);
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
