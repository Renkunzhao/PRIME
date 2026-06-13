///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026 Jiarong Kang
//
// Developed at the Legged AI Lab, University of Wisconsin-Madison.
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>
#include <stdexcept>
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

}  // namespace

int main(int argc, char* argv[]) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: g1_real <config.xml>\n";
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

    crocoddyl::ContactAnitescuIDProblem problem_builder(
        model, joints, cfg.contact_frames, cfg.weights,
        contact_id_experiment::force_log_path(cfg.outputs));

    boost::shared_ptr<crocoddyl::ShootingProblem> shooting_problem =
        problem_builder.createEstimationProblem(
            data.x0, cfg.solver.down_sample * cfg.solver.interval,
            data.state_task.size(), data.state_task, data.ctrl_task);
    shooting_problem->set_nthreads(cfg.solver.n_thread);

    crocoddyl::SolverFDDP solver(shooting_problem);
    solver.set_alphas(make_solver_alphas(cfg.solver.alpha0));
    configure_callbacks(solver, cfg.solver.callbacks);

    crocoddyl::Timer timer;
    solver.solve(data.ctrl_init.empty() ? data.state_task : data.state_init,
                 data.ctrl_init, cfg.solver.max_iter, false, 0.1);
    std::cout << "Duration: " << timer.get_duration() / 1000. << " seconds\n";

    contact_id_experiment::save_solution_outputs(
        cfg.outputs, shooting_problem, solver.get_xs(), solver.get_us());
    contact_id_experiment::save_inertia_identification_report(
        cfg.outputs, model, joints, data.parameter_logs, solver.get_xs(),
        solver.get_us());
  } catch (const std::exception& e) {
    std::cerr << "g1_real: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
