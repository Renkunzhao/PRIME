///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "factory/arm.hpp"

int main(int argc, char *argv[])
{
  bool CALLBACKS = false;
  unsigned int N = 100; // number of nodes
  unsigned int T = 5e3; // number of trials
  unsigned int MAXITER = 1;
  if (argc > 1)
  {
    T = atoi(argv[1]);
  }

  // Building the running and terminal models
  boost::shared_ptr<crocoddyl::ActionModelAbstract> runningModel, terminalModel;
  crocoddyl::benchmark::build_arm_action_models(runningModel, terminalModel);

  // Get the initial state
  boost::shared_ptr<crocoddyl::StateMultibody> state =
      boost::static_pointer_cast<crocoddyl::StateMultibody>(
          runningModel->get_state());
  std::cout << "NQ: " << state->get_nq() << std::endl;
  std::cout << "Number of nodes: " << N << std::endl;
  Eigen::VectorXd q0 =
      state->get_pinocchio()->referenceConfigurations["arm_up"];
  Eigen::VectorXd x0(state->get_nx());
  x0 << q0, Eigen::VectorXd::Random(state->get_nv());

  // For this optimal control problem, we define 100 knots (or running action
  // models) plus a terminal knot
  std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> runningModels(
      N, runningModel);
  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      boost::make_shared<crocoddyl::ShootingProblem>(x0, runningModels,
                                                     terminalModel);
  std::vector<Eigen::VectorXd> xs(N + 1, x0);
  std::vector<Eigen::VectorXd> us(
      N, Eigen::VectorXd::Zero(runningModel->get_nu()));
  for (unsigned int i = 0; i < N; ++i)
  {
    const boost::shared_ptr<crocoddyl::ActionModelAbstract> &model =
        problem->get_runningModels()[i];
    const boost::shared_ptr<crocoddyl::ActionDataAbstract> &data =
        problem->get_runningDatas()[i];
    model->quasiStatic(data, us[i], x0);
  }

  // // Formulating the optimal control problem
  // crocoddyl::SolverFDDP solver(problem);
  // if (CALLBACKS)
  // {
  //   std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> cbs;
  //   cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
  //   solver.setCallbacks(cbs);
  // }

  // // Solving the optimal control problem
  // Eigen::ArrayXd duration(T);
  // for (unsigned int i = 0; i < T; ++i)
  // {
  //   crocoddyl::Timer timer;
  //   solver.solve(xs, us, MAXITER, false, 0.1);
  //   duration[i] = timer.get_duration();
  // }

  // double avrg_duration = duration.sum() / T;
  // double min_duration = duration.minCoeff();
  // double max_duration = duration.maxCoeff();
  // std::cout << "  FDDP.solve [ms]: " << avrg_duration << " (" << min_duration
  //           << "-" << max_duration << ")" << std::endl;

  // // Running calc
  // for (unsigned int i = 0; i < T; ++i)
  // {
  //   crocoddyl::Timer timer;
  //   problem->calc(xs, us);
  //   duration[i] = timer.get_duration();
  // }

  // avrg_duration = duration.sum() / T;
  // min_duration = duration.minCoeff();
  // max_duration = duration.maxCoeff();
  // std::cout << "  ShootingProblem.calc [ms]: " << avrg_duration << " ("
  //           << min_duration << "-" << max_duration << ")" << std::endl;

  // // Running calcDiff
  // for (unsigned int i = 0; i < T; ++i)
  // {
  //   crocoddyl::Timer timer;
  //   problem->calcDiff(xs, us);
  //   duration[i] = timer.get_duration();
  // }

  // avrg_duration = duration.sum() / T;
  // min_duration = duration.minCoeff();
  // max_duration = duration.maxCoeff();
  // std::cout << "  ShootingProblem.calcDiff [ms]: " << avrg_duration << " ("
  //           << min_duration << "-" << max_duration << ")" << std::endl;

  pinocchio::ModelTpl<double> modeld;
  pinocchio::urdf::buildModel(EXAMPLE_ROBOT_DATA_MODEL_DIR
                              "/kinova_description/robots/kinova.urdf",
                              modeld);
  pinocchio::srdf::loadReferenceConfigurations(
      modeld,
      EXAMPLE_ROBOT_DATA_MODEL_DIR "/kinova_description/srdf/kinova.srdf",
      false);

//   // Running rollout
//   const boost::shared_ptr<crocoddyl::ActionDataAbstract> &data = problem->running_datas_[0];
//   problem->running_models_[0]->calc(data, xs[0], us[0]);
//   Eigen::VectorXd x_next = data->xnext;
//   std::cout << "xnext: " << std::endl
//             << data->xnext << std::endl;

//   problem->running_models_[0]->calcDiff(data, xs[0], us[0]);
//   std::cout << "Fx: " << std::endl
//             << data->Fx << std::endl;
//   std::cout << "Fu: " << std::endl
//             << data->Fu << std::endl;

//   // Numerical differentiation check
//   const double epsilon = 1e-10;
//   double test_epsilon = 1e-6;
//   Eigen::MatrixXd F_u_numerical = Eigen::MatrixXd::Zero(modeld.nv * 2, modeld.nv);
//   for (size_t i = 0; i < modeld.nv; ++i)
//   {
//     Eigen::VectorXd x_next_add = Eigen::VectorXd::Zero(modeld.nv + modeld.nq);
//     Eigen::VectorXd u_add = Eigen::VectorXd::Zero(modeld.nv);
//     u_add(i) = epsilon;
//     Eigen::VectorXd u_test = us[0] + u_add;
//     problem->running_models_[0]->calc(data, xs[0], u_test);
//     x_next_add = data->xnext;
//     Eigen::VectorXd v_diff = x_next_add.tail(modeld.nv) - x_next.tail(modeld.nv);

//     Eigen::VectorXd x_diff_tangential = Eigen::VectorXd::Zero(modeld.nv);
//     pinocchio::difference(modeld, x_next.head(modeld.nq), x_next_add.head(modeld.nq), x_diff_tangential);
//     F_u_numerical.block(0, i, modeld.nv, 1) = x_diff_tangential / epsilon;
//     F_u_numerical.block(modeld.nv, i, modeld.nv, 1) = v_diff / epsilon;
//   }
//   std::cout << "F_u_numerical: " << std::endl
//             << F_u_numerical << std::endl;
}
