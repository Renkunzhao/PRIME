///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026 Jiarong Kang
//
// Developed at the Legged AI Lab, University of Wisconsin-Madison.
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <cctype>
#include <iomanip>
#include <iostream>
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

std::string kappa_label(double kappa) {
  std::ostringstream os;
  os << std::setprecision(12) << kappa;
  std::string label = os.str();
  for (std::size_t i = 0; i < label.size(); ++i) {
    if (std::isalnum(static_cast<unsigned char>(label[i]))) {
      continue;
    }
    label[i] = label[i] == '-' ? 'm' : 'p';
  }
  return label;
}

crocoddyl::ContactIDOutputConfig stage_outputs(
    const crocoddyl::ContactIDOutputConfig& base, std::size_t stage_index,
    double kappa) {
  crocoddyl::ContactIDOutputConfig out = base;
  std::ostringstream folder;
  folder << "stage_" << std::setw(2) << std::setfill('0') << stage_index
         << "_kappa_" << kappa_label(kappa);
  out.directory =
      crocoddyl::contact_id_xml::join_path(base.directory, folder.str());
  return out;
}

std::vector<crocoddyl::ContactIDContinuationStage> solve_stages(
    const crocoddyl::ContactIDXMLConfig& cfg) {
  if (cfg.continuation.enabled) {
    return cfg.continuation.stages;
  }

  crocoddyl::ContactIDContinuationStage single;
  single.kappa = cfg.weights.kappa;
  single.max_iter = cfg.solver.max_iter;
  single.has_max_iter = true;
  return std::vector<crocoddyl::ContactIDContinuationStage>(1, single);
}

boost::shared_ptr<crocoddyl::ShootingProblem> create_problem(
    const pinocchio::Model& model,
    const std::vector<pinocchio::JointIndex>& joints,
    const std::vector<std::string>& contact_frames,
    const crocoddyl::ContactIDWeights& weights,
    const crocoddyl::ContactIDSolverConfig& solver_cfg,
    const crocoddyl::ContactIDOutputConfig& outputs,
    contact_id_experiment::PreparedContactIDData& data) {
  crocoddyl::ContactAnitescuIDProblem problem_builder(
      model, joints, contact_frames, weights,
      contact_id_experiment::force_log_path(outputs));

  boost::shared_ptr<crocoddyl::ShootingProblem> shooting_problem =
      problem_builder.createEstimationProblem(
          data.x0, solver_cfg.down_sample * solver_cfg.interval,
          data.state_task.size(), data.state_task, data.ctrl_task);
  shooting_problem->set_nthreads(solver_cfg.n_thread);
  return shooting_problem;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: go2_real_belly_plate_4p6kg <config.xml>\n";
      return 2;
    }

    const crocoddyl::ContactIDXMLConfig cfg =
        crocoddyl::contact_id_xml::load_config(argv[1]);
    contact_id_experiment::validate_required(cfg);

    if (cfg.solver.dry_run) {
      std::cout << "Parsed contact-ID XML successfully.\n";
      std::cout << "contacts=" << cfg.contact_frames.size()
                << " identified_entries="
                << (cfg.identified_joint_indices.size() +
                    cfg.identified_joint_names.size())
                << "\n";
      const std::vector<crocoddyl::ContactIDContinuationStage> stages =
          solve_stages(cfg);
      std::cout << "kappa_stages=" << stages.size() << "\n";
      for (std::size_t i = 0; i < stages.size(); ++i) {
        const std::size_t max_iter =
            stages[i].has_max_iter ? stages[i].max_iter : cfg.solver.max_iter;
        std::cout << "  stage=" << i << " kappa=" << stages[i].kappa
                  << " max_iter=" << max_iter << "\n";
      }
      return 0;
    }

    const pinocchio::Model model =
        contact_id_experiment::load_floating_base_model(cfg.robot);
    const std::vector<pinocchio::JointIndex> joints =
        contact_id_experiment::resolve_identified_joints(model, cfg);

    contact_id_experiment::PreparedContactIDData data =
        contact_id_experiment::prepare_contact_id_data(
            model, joints, cfg.contact_frames, cfg);

    contact_id_experiment::clear_outputs(cfg.outputs);
    contact_id_experiment::log_initial_guess(cfg.outputs, data.state_task,
                                             data.ctrl_task);

    crocoddyl::Timer timer;
    std::vector<Eigen::VectorXd> xs_guess =
        data.ctrl_init.empty() ? data.state_task : data.state_init;
    std::vector<Eigen::VectorXd> us_guess = data.ctrl_init;
    std::vector<Eigen::VectorXd> final_xs;
    std::vector<Eigen::VectorXd> final_us;

    const std::vector<crocoddyl::ContactIDContinuationStage> stages =
        solve_stages(cfg);
    for (std::size_t i = 0; i < stages.size(); ++i) {
      const double kappa = stages[i].kappa;
      const std::size_t max_iter =
          stages[i].has_max_iter ? stages[i].max_iter : cfg.solver.max_iter;

      crocoddyl::ContactIDWeights stage_weights = cfg.weights;
      stage_weights.kappa = kappa;
      const crocoddyl::ContactIDOutputConfig outputs =
          cfg.continuation.enabled ? stage_outputs(cfg.outputs, i, kappa)
                                   : cfg.outputs;
      contact_id_experiment::clear_outputs(outputs);
      contact_id_experiment::log_initial_guess(outputs, data.state_task,
                                               data.ctrl_task);
      contact_id_experiment::save_vector_sequence_csv(
          outputs, "xs_warm_start.csv", xs_guess);
      contact_id_experiment::save_vector_sequence_csv(
          outputs, "us_warm_start.csv", us_guess);

      std::cout << "Solving stage " << i << " / " << stages.size()
                << " with kappa=" << kappa << " max_iter=" << max_iter
                << "\n";
      boost::shared_ptr<crocoddyl::ShootingProblem> shooting_problem =
          create_problem(model, joints, cfg.contact_frames, stage_weights,
                         cfg.solver, outputs, data);

      crocoddyl::SolverFDDP solver(shooting_problem);
      solver.set_alphas(make_solver_alphas(cfg.solver.alpha0));
      configure_callbacks(solver, cfg.solver.callbacks);
      solver.solve(xs_guess, us_guess, max_iter, false, 0.1);

      final_xs = solver.get_xs();
      final_us = solver.get_us();
      contact_id_experiment::save_solution_outputs(
          outputs, shooting_problem, final_xs, final_us);
      contact_id_experiment::save_inertia_identification_report(
          outputs, model, joints, data.parameter_logs, final_xs, final_us);

      xs_guess = final_xs;
      us_guess = final_us;
    }
    std::cout << "Duration: " << timer.get_duration() / 1000. << " seconds\n";

    if (cfg.continuation.enabled && !final_xs.empty()) {
      crocoddyl::ContactIDWeights final_weights = cfg.weights;
      final_weights.kappa = stages.back().kappa;
      boost::shared_ptr<crocoddyl::ShootingProblem> final_problem =
          create_problem(model, joints, cfg.contact_frames, final_weights,
                         cfg.solver, cfg.outputs, data);
      contact_id_experiment::save_solution_outputs(cfg.outputs, final_problem,
                                                   final_xs, final_us);
      contact_id_experiment::save_inertia_identification_report(
          cfg.outputs, model, joints, data.parameter_logs, final_xs, final_us);
    }
  } catch (const std::exception& e) {
    std::cerr << "go2_real_belly_plate_4p6kg: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
