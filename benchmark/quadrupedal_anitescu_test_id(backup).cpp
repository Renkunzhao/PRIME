#include <example-robot-data/path.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/regressor.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/spatial/inertia.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>

#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <iostream>
#include <vector>
#include <string>

// Your state + action headers
#include "crocoddyl/multibody/states/multibody_params.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/integrator/euler.hpp"

#include "crocoddyl/multibody/actions/contact-fwddyn-anitescu-id.hpp"
#include "crocoddyl/multibody/actuations/floating-base-estimation.hpp"

static inline double relError(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
{
  const double denom = std::max(1.0, std::max(A.norm(), B.norm()));
  return (A - B).norm() / denom;
}

int main()
{
  using Scalar = double;

  // ---------- Build model ----------
  pinocchio::Model model;
  pinocchio::urdf::buildModel(
      std::string(EXAMPLE_ROBOT_DATA_MODEL_DIR) + "/go1_description/urdf/go1.urdf",
      pinocchio::JointModelFreeFlyer(), model);

  Eigen::MatrixXd xs_result = csvutil::readCSVtoEigen("data_go1_rl/xs_results_fddp.csv");
  Eigen::MatrixXd us_result = csvutil::readCSVtoEigen("data_go1_rl/us_results_fddp.csv");
  std::vector<Eigen::VectorXd> state_stack;
  std::vector<Eigen::VectorXd> ctrl_stack;
  std::size_t N = xs_result.rows() - 10;

  for (std::size_t i = 0; i < N + 1; i += 1)
  {
    Eigen::VectorXd xs_i = xs_result.row(i).transpose();
    xs_i.segment<4>(3).normalize();
    state_stack.push_back(xs_i);
  }

  for (std::size_t i = 0; i < N; i += 1)
  {
    Eigen::VectorXd ctrl_i = us_result.row(i).segment(0, 18).transpose();
    ctrl_stack.push_back(ctrl_i);
  }

  // ---------- Build StateMultibodyParams ----------
  pinocchio::JointIndex j_joint = 1;
  std::vector<pinocchio::JointIndex> joints;
  joints.push_back(j_joint);

  Eigen::VectorXd pi_log = computeLogCholeskyFromLink(model, j_joint);
  pinocchio::LogCholeskyParametersTpl<double> log_chol(pi_log);
  std::cout << log_chol.toPseudoInertia() << std::endl;
  boost::shared_ptr<pinocchio::ModelTpl<double>> model_ptr = boost::make_shared<pinocchio::ModelTpl<double>>(model);

  auto state_id = boost::make_shared<crocoddyl::StateMultibodyParams>(model_ptr, joints);

  // ---------- Build Actuation ----------
  auto actuation = boost::make_shared<crocoddyl::ActuationModelFloatingBaseEstimation>(state_id);

  // // ---------- Cost (can be empty/zero) ----------
  auto cost = boost::make_shared<crocoddyl::CostModelSum>(state_id, actuation->get_nu());

  // ---------- Build your ID differential model ----------
  using DAM_ID = crocoddyl::DifferentialActionModelContactFwdDynamicsAnitescuSystemidTpl<double>;
  using DAD_ID = crocoddyl::DifferentialActionDataContactFwdDynamicsAnitescuSystemidTpl<double>;

  const double dt = 0.02;
  std::vector<std::string> foot_names = {"RR_foot_fixed", "RL_foot_fixed", "FR_foot_fixed", "FL_foot_fixed"};

  auto dmodel = boost::make_shared<DAM_ID>(
      state_id,
      actuation,
      cost,
      foot_names,
      40,
      0.7,
      dt);

  // auto imodel = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel, dt);

  // ---------- Allocate data ----------
  auto ddata = dmodel->createData();
  // auto idata = imodel->createData();

  // ---------- Evaluate ----------
  // Eigen::VectorXd x = Eigen::VectorXd::Zero(model.nq + model.nv + 10 + 10);
  // x.segment(0, model.nq) = state_stack[0].segment(0, model.nq);
  // x.segment(model.nq, 10) = pi_log;
  // x.segment(model.nq + 10, model.nv) = state_stack[0].segment(model.nq, model.nv);

  // Eigen::VectorXd u = ctrl_stack[0];
  // dmodel->calc(ddata, x, u);
  // dmodel->calcDiff(ddata, x, u);

  // auto *d = static_cast<DAD_ID *>(ddata.get());
  // std::cout << d->f_contact_constr << std::endl;
  // std::cout << d->df_dx << std::endl;
  // std::cout << d->xout << std::endl;
  // std::cout << d->Fx.block(0, model.nv, model.nv, 10) << std::endl;
  // std::cout << d->Fu << std::endl;

  // // Central Different check on forward dynamics differentiation wrsp to params
  // //--------------------------------------------------------------------------------------------
  const double eps = 1e-6;

  // for (std::size_t idx = 0; idx < 10; ++idx)
  // {
  //   Eigen::VectorXd x = Eigen::VectorXd::Zero(model.nq + model.nv + 10 + 10);
  //   x.segment(0, model.nq) = state_stack[idx].segment(0, model.nq);
  //   // x[2] = x[2] + 10;
  //   x.segment(model.nq, 10) = pi_log;
  //   x.segment(model.nq + 10, model.nv) = state_stack[idx].segment(model.nq, model.nv);

  //   Eigen::VectorXd u = ctrl_stack[idx];

  //   dmodel->calc(ddata, x, u);
  //   dmodel->calcDiff(ddata, x, u);
  //   auto *d0 = static_cast<DAD_ID *>(ddata.get());

  //   const Eigen::VectorXd y0 = d0->xout;
  //   const int ny = static_cast<int>(y0.size());

  //   Eigen::MatrixXd Jy_num(ny, 10);
  //   Jy_num.setZero();

  //   for (int k = 0; k < 10; ++k)
  //   {
  //     Eigen::VectorXd x_plus = x;
  //     Eigen::VectorXd x_minus = x;

  //     x_plus(model.nq + k) += eps;
  //     x_minus(model.nq + k) -= eps;

  //     // +eps
  //     auto data_p = dmodel->createData();
  //     dmodel->calc(data_p, x_plus, u);
  //     dmodel->calcDiff(data_p, x_plus, u);
  //     auto *dp = static_cast<DAD_ID *>(data_p.get());
  //     const Eigen::VectorXd y_plus = dp->xout;

  //     // -eps
  //     auto data_m = dmodel->createData();
  //     dmodel->calc(data_m, x_minus, u);
  //     dmodel->calcDiff(data_m, x_minus, u);
  //     auto *dm = static_cast<DAD_ID *>(data_m.get());
  //     const Eigen::VectorXd y_minus = dm->xout;

  //     Jy_num.col(k) = (y_plus - y_minus) / (2.0 * eps);
  //   }

  //   Eigen::MatrixXd Jy_an = d0->Fx.block(0, model.nv, ny, 10);

  //   const Eigen::MatrixXd E = Jy_an - Jy_num;
  //   std::cout << "Analytic Jy (from Fx block):\n"
  //             << Jy_an << "\n\n";
  //   std::cout << "Numeric Jy (central diff):\n"
  //             << Jy_num << "\n\n";
  //   std::cout << "Error J:\n"
  //             << E << "\n\n";
  //   std::cout << "Max abs error: " << E.cwiseAbs().maxCoeff() << "\n";
  //   Eigen::Index r_max = -1, c_max = -1;
  //   const double e_max = E.cwiseAbs().maxCoeff(&r_max, &c_max);

  //   std::cout << "Max abs error: " << e_max
  //             << " at (row=" << r_max << ", col=" << c_max << ")\n";
  //   std::cout << "  Analytic Jy:  " << Jy_an(r_max, c_max) << "\n";
  //   std::cout << "  Numeric  Jy:  " << Jy_num(r_max, c_max) << "\n";
  //   std::cout << "  Diff (an-num): " << E(r_max, c_max) << "\n";
  // }

  // Central Different check on forward dynamics differentiation wrsp to velocity
  //--------------------------------------------------------------------------------------------
  // for (std::size_t idx = 0; idx < 10; ++idx)
  // {
  //   Eigen::VectorXd x = Eigen::VectorXd::Zero(model.nq + model.nv + 10 + 10);
  //   x.segment(0, model.nq) = state_stack[idx].segment(0, model.nq);
  //   x.segment(model.nq, 10) = pi_log;
  //   x.segment(model.nq + 10, model.nv) = state_stack[idx].segment(model.nq, model.nv);

  //   Eigen::VectorXd u = ctrl_stack[idx];

  //   // baseline
  //   dmodel->calc(ddata, x, u);
  //   dmodel->calcDiff(ddata, x, u);
  //   auto *d0 = static_cast<DAD_ID *>(ddata.get());

  //   const Eigen::VectorXd y0 = d0->xout;
  //   // const Eigen::VectorXd y0 = d0->f_contact_constr;

  //   const int ny = static_cast<int>(y0.size());

  //   // Numeric Jacobian wrt dq (tangent, dim = nv)
  //   Eigen::MatrixXd Jy_num(ny, model.nv);
  //   Jy_num.setZero();

  //   // Extract current q as a concrete vector (avoid aliasing issues with Eigen::Ref)
  //   const Eigen::VectorXd q = x.head(model.nq).eval();

  //   for (int k = 0; k < static_cast<int>(model.nv); ++k)
  //   {
  //     // tangent perturbation
  //     Eigen::VectorXd dq_plus = Eigen::VectorXd::Zero(model.nv);
  //     Eigen::VectorXd dq_minus = Eigen::VectorXd::Zero(model.nv);
  //     dq_plus(k) = +eps;
  //     dq_minus(k) = -eps;

  //     // integrate on configuration manifold
  //     const Eigen::VectorXd q_plus = pinocchio::integrate(model, q, dq_plus);
  //     const Eigen::VectorXd q_minus = pinocchio::integrate(model, q, dq_minus);

  //     Eigen::VectorXd x_plus = x;
  //     Eigen::VectorXd x_minus = x;
  //     x_plus.head(model.nq) = q_plus;
  //     x_minus.head(model.nq) = q_minus;

  //     // +eps
  //     auto data_p = dmodel->createData();
  //     dmodel->calc(data_p, x_plus, u);
  //     dmodel->calcDiff(data_p, x_plus, u);

  //     auto *dp = static_cast<DAD_ID *>(data_p.get());
  //     // const Eigen::VectorXd y_plus = dp->f_contact_constr;
  //     const Eigen::VectorXd y_plus = dp->xout;

  //     // -eps
  //     auto data_m = dmodel->createData();
  //     dmodel->calc(data_m, x_minus, u);
  //     dmodel->calcDiff(data_m, x_minus, u);
  //     auto *dm = static_cast<DAD_ID *>(data_m.get());
  //     // const Eigen::VectorXd y_minus = dm->f_contact_constr;
  //     const Eigen::VectorXd y_minus = dm->xout;

  //     Jy_num.col(k) = (y_plus - y_minus) / (2.0 * eps);
  //   }

  //   // Analytic dq block (assuming dx ordering: [dq (nv), dp (10), dv (nv), ...])
  //   Eigen::MatrixXd Jy_an = d0->Fx.block(0, 0, ny, model.nv);
  //   // Eigen::MatrixXd Jy_an = d0->df_dx.block(0, 0, ny, model.nv);

  //   const Eigen::MatrixXd E = Jy_an - Jy_num;

  //   Eigen::Index r_max = -1, c_max = -1;
  //   const double e_max = E.cwiseAbs().maxCoeff(&r_max, &c_max);

  //   std::cout << "Analytic Jy (dq):\n"
  //             << Jy_an << "\n\n";
  //   std::cout << "Numeric Jy (dq, manifold integrate):\n"
  //             << Jy_num << "\n\n";
  //   std::cout << "Error J:\n"
  //             << E << "\n\n";

  //   std::cout << "Max abs error: " << e_max
  //             << " at (row=" << r_max << ", col=" << c_max << ")\n";
  //   std::cout << "  Analytic Jy:   " << Jy_an(r_max, c_max) << "\n";
  //   std::cout << "  Numeric  Jy:   " << Jy_num(r_max, c_max) << "\n";
  //   std::cout << "  Diff (an-num): " << E(r_max, c_max) << "\n";
  // }

  // // // Central Different check on force dynamics differentiation wrsp to params
  // // //--------------------------------------------------------------------------------------------
  // // // const double eps = 1e-6;
  // // for (std::size_t idx = 0; idx < 5; ++idx)
  // // {
  // //   Eigen::VectorXd x = Eigen::VectorXd::Zero(model.nq + model.nv + 10 + 10);
  // //   x.segment(0, model.nq) = state_stack[idx].segment(0, model.nq);
  // //   x.segment(model.nq, 10) = pi_log;
  // //   x.segment(model.nq + 10, model.nv) = state_stack[idx].segment(model.nq, model.nv);

  // //   Eigen::VectorXd u = ctrl_stack[idx];

  // //   dmodel->calc(ddata, x, u);
  // //   dmodel->calcDiff(ddata, x, u);
  // //   auto *d0 = static_cast<DAD_ID *>(ddata.get());

  // //   const Eigen::VectorXd y0 = d0->f_contact_constr; // size likely nv
  // //   const int ny = static_cast<int>(y0.size());

  // //   Eigen::MatrixXd Jy_num(ny, 10);
  // //   Jy_num.setZero();

  // //   for (int k = 0; k < 10; ++k)
  // //   {
  // //     Eigen::VectorXd x_plus = x;
  // //     Eigen::VectorXd x_minus = x;

  // //     x_plus(model.nq + k) += eps; // pi_log lives here in your x layout
  // //     x_minus(model.nq + k) -= eps;

  // //     // +eps
  // //     auto data_p = dmodel->createData();
  // //     dmodel->calc(data_p, x_plus, u);
  // //     dmodel->calcDiff(data_p, x_plus, u);
  // //     auto *dp = static_cast<DAD_ID *>(data_p.get());
  // //     const Eigen::VectorXd y_plus = dp->f_contact_constr;

  // //     // -eps
  // //     auto data_m = dmodel->createData();
  // //     dmodel->calc(data_m, x_minus, u);
  // //     dmodel->calcDiff(data_m, x_minus, u);
  // //     auto *dm = static_cast<DAD_ID *>(data_m.get());
  // //     const Eigen::VectorXd y_minus = dm->f_contact_constr;

  // //     Jy_num.col(k) = (y_plus - y_minus) / (2.0 * eps);
  // //   }

  // //   Eigen::MatrixXd Jy_an = d0->df_dx.block(0, model.nv, ny, 10);

  // //   const Eigen::MatrixXd E = Jy_an - Jy_num;
  // //   std::cout << "Analytic Jy (from Fx block):\n"
  // //             << Jy_an << "\n\n";
  // //   std::cout << "Numeric Jy (central diff):\n"
  // //             << Jy_num << "\n\n";
  // //   std::cout << "Max abs error: " << E.cwiseAbs().maxCoeff() << "\n";
  // //   std::cout << "Rel error (Frob): "
  // //             << E.norm() / std::max(1.0, Jy_num.norm()) << "\n";
  // // }

  boost::shared_ptr<crocoddyl::IntegratedActionModelAbstract> imodel_euler =
      boost::make_shared<crocoddyl::IntegratedActionModelEuler>(dmodel,
                                                                dt);

  auto data_e = imodel_euler->createData();

  // for (int idx = 0; idx < 5; ++idx)
  // {
  //   Eigen::VectorXd x = Eigen::VectorXd::Zero(model.nq + model.nv + 10 + 10);
  //   x.segment(0, model.nq) = state_stack[idx].segment(0, model.nq);
  //   x.segment(model.nq, 10) = pi_log;
  //   x.segment(model.nq + 10, model.nv) = state_stack[idx].segment(model.nq, model.nv);
  //   Eigen::VectorXd u = ctrl_stack[idx];

  //   imodel_euler->calc(data_e, x, u);
  //   imodel_euler->calcDiff(data_e, x, u);

  //   crocoddyl::IntegratedActionDataEuler *de =
  //       static_cast<crocoddyl::IntegratedActionDataEuler *>(
  //           data_e.get());

  //   std::cout << "Fx" << std::endl
  //             << de->Fx << std::endl;
  //   std::cout << "Fu" << std::endl
  //             << de->Fu << std::endl;
  // }

  auto state = imodel_euler->get_state();

  for (int idx = 0; idx < 20; ++idx)
  {
    Eigen::VectorXd x = Eigen::VectorXd::Zero(model.nq + model.nv + 10 + 10);
    x.segment(0, model.nq) = state_stack[idx].segment(0, model.nq);
    x.segment(model.nq, 10) = pi_log;
    x.segment(model.nq + 10, model.nv) = state_stack[idx].segment(model.nq, model.nv);
    Eigen::VectorXd u = ctrl_stack[idx];

    // --- baseline ---
    imodel_euler->calc(data_e, x, u);
    imodel_euler->calcDiff(data_e, x, u);

    auto *de = static_cast<crocoddyl::IntegratedActionDataEuler *>(data_e.get());

    const Eigen::MatrixXd Fx_an = de->Fx;
    const Eigen::MatrixXd Fu_an = de->Fu;

    // Choose what "output" to differentiate.
    // For integrated action model, the output corresponds to data->xnext in Crocoddyl.
    // In Euler data, this is usually: de->xnext (or de->xout depending on your branch).
    const Eigen::VectorXd y0 = de->xnext; // <-- if your struct uses xnext
    const int ny = (int)y0.size() - 1;

    // --- numeric Fx ---
    Eigen::MatrixXd Fx_num(ny, state->get_ndx());
    Fx_num.setZero();

    for (int k = 0; k < (int)state->get_ndx(); ++k)
    {
      Eigen::VectorXd dx = Eigen::VectorXd::Zero(state->get_ndx());
      dx(k) = eps;

      Eigen::VectorXd x_plus(state->get_nx()), x_minus(state->get_nx());
      state->integrate(x, dx, x_plus);
      state->integrate(x, -dx, x_minus);
      state->diff(x_minus, x_plus, dx);
      Eigen::VectorXd dxx = dx / (2 * eps);
      // std::cout << "changed" << dxx.transpose() << std::endl;
      auto dp = imodel_euler->createData();
      imodel_euler->calc(dp, x_plus, u);
      auto *dep = static_cast<crocoddyl::IntegratedActionDataEuler *>(dp.get());
      Eigen::VectorXd y_plus = dep->xnext;

      auto dm = imodel_euler->createData();
      imodel_euler->calc(dm, x_minus, u);
      auto *dem = static_cast<crocoddyl::IntegratedActionDataEuler *>(dm.get());
      Eigen::VectorXd y_minus = dem->xnext;

      Eigen::VectorXd dy = Eigen::VectorXd::Zero(state->get_ndx());
      state->diff(y_minus, y_plus, dy);
      Fx_num.col(k) = dy / (2.0 * eps);
    }

    // --- numeric Fu ---
    Eigen::MatrixXd Fu_num(ny, (int)u.size());
    Fu_num.setZero();

    for (int k = 0; k < (int)u.size(); ++k)
    {
      Eigen::VectorXd u_plus = u, u_minus = u;
      u_plus(k) += eps;
      u_minus(k) -= eps;

      auto dp = imodel_euler->createData();
      imodel_euler->calc(dp, x, u_plus);
      auto *dep = static_cast<crocoddyl::IntegratedActionDataEuler *>(dp.get());
      Eigen::VectorXd y_plus = dep->xnext;

      auto dm = imodel_euler->createData();
      imodel_euler->calc(dm, x, u_minus);
      auto *dem = static_cast<crocoddyl::IntegratedActionDataEuler *>(dm.get());
      Eigen::VectorXd y_minus = dem->xnext;

      Eigen::VectorXd dy = Eigen::VectorXd::Zero(state->get_ndx());
      state->diff(y_minus, y_plus, dy);
      Fu_num.col(k) = dy / (2.0 * eps);
    }

    // --- Compare Fx ---
    // Eigen::MatrixXd Ex = Fx_an.topRows(ny) - Fx_num; // guard if Fx has extra rows
    // Eigen::Index rx = -1, cx = -1;
    // double emax_x = Ex.cwiseAbs().maxCoeff(&rx, &cx);

    // std::cout << Fx_an.block(0, 18 + 10, 18 + 10, 18 + 10) << std::endl;
    // std::cout << Fx_num.block(0, 18 + 10, 18 + 10, 18 + 10) << std::endl;

    // std::cout << "\n[idx " << idx << "] Fx check\n";
    // std::cout << "Max abs err Fx: " << emax_x << " at (" << rx << "," << cx << ")\n";
    // std::cout << "  Fx_an: " << Fx_an(rx, cx) << "\n";
    // std::cout << "  Fx_num:" << Fx_num(rx, cx) << "\n";
    // std::cout << "  diff: " << (Fx_an(rx, cx) - Fx_num(rx, cx)) << "\n";
    // std::cout << "Rel Fro err Fx: " << Ex.norm() / std::max(1.0, Fx_num.norm()) << "\n";

    // --- Compare Fu ---
    Eigen::MatrixXd Eu = Fu_an.topRows(ny) - Fu_num;
    Eigen::Index ru = -1, cu = -1;
    double emax_u = Eu.cwiseAbs().maxCoeff(&ru, &cu);

    std::cout << "\n-----------------\n";
    std::cout << Fu_an.block(0, 0, ny, 18) << std::endl;
    std::cout << "\n-----------------\n";
    std::cout << Fu_num.block(0, 0, ny, 18) << std::endl;
    std::cout << "\n[idx " << idx << "] Fu check\n";
    std::cout << "Max abs err Fu: " << emax_u << " at (" << ru << "," << cu << ")\n";
    std::cout << "  Fu_an: " << Fu_an(ru, cu) << "\n";
    std::cout << "  Fu_num:" << Fu_num(ru, cu) << "\n";
    std::cout << "  diff: " << (Fu_an(ru, cu) - Fu_num(ru, cu)) << "\n";
    std::cout << "Rel Fro err Fu: " << Eu.norm() / std::max(1.0, Fu_num.norm()) << "\n";
  }

  for (const auto jid : joints)
  {
    std::cout << "jid=" << jid << " name=" << model.names[jid] << "\n";
  }
  std::cout << pi_log.transpose() << std::endl;

  return 0;
}