///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <example-robot-data/path.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>

#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "crocoddyl/multibody/utils/quadruped-anitescu-id.hpp"
#include "crocoddyl/multibody/utils/csv_to_eigen.hpp"
#include <random>
#include <filesystem>
#include "crocoddyl/multibody/utils/rapidxml.hpp"
#include <pinocchio/spatial/inertia.hpp> // PseudoInertia, Inertia

using namespace rapidxml;

// First-order low-pass (exponential smoothing) applied along time (rows).
// xs: N x nv (N timesteps, nv dims). Each column filtered independently.
// y[0] = x[0]
// y[t] = y[t-1] + alpha * (x[t] - y[t-1])
inline Eigen::MatrixXd lowpass_first_order_alpha(
    const Eigen::Ref<const Eigen::MatrixXd> &xs,
    double alpha)
{
  if (xs.rows() == 0)
    return Eigen::MatrixXd();
  if (!(alpha > 0.0 && alpha <= 1.0))
  {
    throw std::invalid_argument("alpha must be in (0, 1].");
  }

  Eigen::MatrixXd ys(xs.rows(), xs.cols());
  ys.row(0) = xs.row(0);

  for (Eigen::Index t = 1; t < xs.rows(); ++t)
  {
    // vectorized per-row update
    ys.row(t) = ys.row(t - 1) + alpha * (xs.row(t) - ys.row(t - 1));
  }
  return ys;
}

// Convenience wrapper: specify cutoff frequency fc (Hz) and dt (s).
// alpha = dt / (tau + dt), tau = 1/(2*pi*fc)
inline Eigen::MatrixXd lowpass_first_order_cutoff(
    const Eigen::Ref<const Eigen::MatrixXd> &xs,
    double dt,
    double fc_hz)
{
  if (!(dt > 0.0))
  {
    throw std::invalid_argument("dt must be > 0.");
  }
  if (!(fc_hz > 0.0))
  {
    throw std::invalid_argument("fc_hz must be > 0.");
  }

  const double tau = 1.0 / (2.0 * M_PI * fc_hz);
  const double alpha = dt / (tau + dt);
  return lowpass_first_order_alpha(xs, alpha);
}

void printPinocchioIndexOrder(const pinocchio::Model &model)
{
  std::cout << "================ Pinocchio Model ================\n";
  std::cout << "name        : " << model.name << "\n";
  std::cout << "nq, nv      : " << model.nq << ", " << model.nv << "\n";
  std::cout << "#joints(fr) : " << model.njoints << " joints, "
            << model.nframes << " frames\n\n";

  std::cout << "---- Joint index order (q/v layout) ----\n";
  // Joint 0 is "universe" (no dofs); start from 1
  for (pinocchio::JointIndex jid = 1; jid < model.njoints; ++jid)
  {
    const auto &j = model.joints[jid];
    const std::string &jn = model.names[jid];
    const auto parent = model.parents[jid];
    const std::string parent_name = (parent < model.names.size() ? model.names[parent] : "N/A");

    std::cout << std::left << std::setw(20) << jn
              << "  q:[" << std::setw(2) << j.idx_q() << " .. "
              << (j.idx_q() + j.nq() - 1) << "] (nq=" << j.nq() << ")"
              << "  v:[" << std::setw(2) << j.idx_v() << " .. "
              << (j.idx_v() + j.nv() - 1) << "] (nv=" << j.nv() << ")"
              << "  parent=" << parent_name
              << "\n";
    std::cout << model.inertias[jid].toDynamicParameters().transpose() << "\n";
  }
  std::cout << "\n";
}

double computeLowestFootHeight(const pinocchio::Model &model,
                               pinocchio::Data &data,
                               const Eigen::VectorXd &q)
{
  // 1) Forward kinematics & update all frame placements
  pinocchio::forwardKinematics(model, data, q);
  pinocchio::updateFramePlacements(model, data);

  // 2) Foot frame names in the model
  std::vector<std::string> foot_frames = {
      "LR_FOOT_FL", "LR_FOOT_FR", "LR_FOOT_RL", "LR_FOOT_RR",
      "LL_FOOT_FL", "LL_FOOT_FR", "LL_FOOT_RL", "LL_FOOT_RR"};

  double min_z = std::numeric_limits<double>::infinity();

  for (const auto &name : foot_frames)
  {
    // Get frame ID
    pinocchio::FrameIndex fid = model.getFrameId(name);
    if (fid == (pinocchio::FrameIndex)(-1))
    {
      std::cerr << "Frame \"" << name << "\" not found in model.\n";
      continue;
    }

    // World placement of this frame: oMf[fid]
    const pinocchio::SE3 &oMf = data.oMf[fid];
    const Eigen::Vector3d &p_world = oMf.translation();

    double z = p_world.z(); // or p_world(2)

    // std::cout << "Frame " << name
    //           << " position in world: [" << p_world.transpose() << "]\n";

    if (z < min_z)
      min_z = z;
  }

  return min_z;
}

// Eigen::Matrix4d computeUpperUU(const Eigen::Matrix4d &J)
// {
//   // permutation that reverses order: (0,1,2,3) -> (3,2,1,0)
//   Eigen::Matrix4d P = Eigen::Matrix4d::Zero();
//   for (int i = 0; i < 4; ++i)
//     P(i, 3 - i) = 1.0;

//   Eigen::Matrix4d J_hat = P * J * P.transpose();

//   Eigen::LLT<Eigen::Matrix4d> llt(J_hat);
//   if (llt.info() != Eigen::Success)
//     throw std::runtime_error("Cholesky failed, J not SPD");

//   Eigen::Matrix4d L_hat = llt.matrixL();         // lower-triangular
//   Eigen::Matrix4d U = P.transpose() * L_hat * P; // now upper-triangular

//   // Sanity check: J ≈ U * U^T
//   // Eigen::Matrix4d J_rec = U * U.transpose();

//   return U;
// }

// // pinocchio::LogCholeskyParametersTpl<double>
// Eigen::Matrix<double, 10, 1>
// computeLogCholeskyFromLink(const pinocchio::Model &model, int j_link)
// {
//   // 1) Get inertia of the link and convert to pseudo-inertia
//   const pinocchio::Inertia &Ij = model.inertias[j_link];
//   pinocchio::PseudoInertia Ji = Ij.toPseudoInertia();
//   Eigen::Matrix4d Ji_mat = Ji.toMatrix();
//   std::cout << "PseudoInertia from model: \n"
//             << Ji_mat << "\n";
//   Eigen::Matrix4d R = computeUpperUU(Ji_mat); // upper-triangular factor

//   // 3) Extract global scale alpha and normalize so R_scaled(3,3) = 1
//   const double alpha = std::log(R(3, 3));
//   Eigen::Matrix4d R_scaled = R / R(3, 3);

//   // 4) Fill log–Cholesky parameter vector (same ordering as Pinocchio)
//   // parameters: [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]
//   Eigen::Matrix<double, 10, 1> pi_log_chol;

//   pi_log_chol(0) = alpha;                    // alpha
//   pi_log_chol(1) = std::log(R_scaled(0, 0)); // d1
//   pi_log_chol(2) = std::log(R_scaled(1, 1)); // d2
//   pi_log_chol(3) = std::log(R_scaled(2, 2)); // d3

//   pi_log_chol(4) = R_scaled(0, 1); // s12
//   pi_log_chol(5) = R_scaled(1, 2); // s23
//   pi_log_chol(6) = R_scaled(0, 2); // s13

//   pi_log_chol(7) = R_scaled(0, 3); // t1
//   pi_log_chol(8) = R_scaled(1, 3); // t2
//   pi_log_chol(9) = R_scaled(2, 3); // t3

//   std::cout << pi_log_chol << std::endl;
//   // 5) Construct and return Pinocchio LogCholeskyParameters object
//   // return pinocchio::LogCholeskyParametersTpl<double>(pi_log_chol);
//   return pi_log_chol;
// }

Eigen::VectorXd generateGaussianNoise(int n, double stddev = 1.0)
{
  Eigen::VectorXd noise(n);

  std::default_random_engine generator(std::random_device{}());
  std::normal_distribution<double> distribution(0.0, stddev); // mean = 0

  for (int i = 0; i < n; ++i)
  {
    noise[i] = distribution(generator);
  }

  return noise;
}

// qs row format (Pinocchio FreeFlyer):
// [x y z qx qy qz qw | joints...]
// Here, qs base is WORLD->TORSO (not pelvis).

// Build SE3 from (x,y,z,qx,qy,qz,qw) where quat is xyzw.
static inline pinocchio::SE3 se3FromXYZQuat_xyzw(const Eigen::Vector3d &p,
                                                 const Eigen::Vector4d &q_xyzw)
{
  Eigen::Quaterniond q(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]); // Eigen ctor is (w,x,y,z)
  q.normalize();
  return pinocchio::SE3(q.toRotationMatrix(), p);
}

// Convert SE3 rotation to (qx,qy,qz,qw) (xyzw).
static inline Eigen::Vector4d quat_xyzw_FromSE3(const pinocchio::SE3 &M)
{
  Eigen::Quaterniond q(M.rotation());
  q.normalize();
  return Eigen::Vector4d(q.x(), q.y(), q.z(), q.w());
}

/**
 * Input:
 *   qs: N x model.nq
 *       qs[k,0:7] is WORLD->TORSO pose (xyzw),
 *       qs[k,7:]  is joint configuration (same ordering as the pelvis-root model expects).
 *
 * Output:
 *   pelvis_world: N x 7, WORLD->PELVIS pose (xyzw).
 *
 * Model requirement:
 *   buildModel(urdf, JointModelFreeFlyer(), model) where the URDF root link is pelvis.
 */
Eigen::MatrixXd computePelvisWorldFromTorsoBase_xyzw(
    const pinocchio::Model &model,
    const std::string &torso_link_name,
    const Eigen::Ref<const Eigen::MatrixXd> &qs)
{
  if (model.nq < 7)
    throw std::runtime_error("Expected a FreeFlyer model (nq >= 7).");
  if (qs.cols() != model.nq)
    throw std::runtime_error("qs must be N x model.nq.");

  const int N = static_cast<int>(qs.rows());
  Eigen::MatrixXd pelvis_world(N, 7);

  pinocchio::Data data(model);
  const pinocchio::FrameIndex torso_fid = model.getFrameId(torso_link_name, pinocchio::BODY);

  // FK config: pelvis free-flyer = identity, joints from qs
  Eigen::VectorXd q_fk(model.nq);
  q_fk.setZero();
  // identity free-flyer in Pinocchio layout: [0 0 0  0 0 0 1]
  q_fk.segment<3>(0).setZero();
  q_fk.segment<4>(3) << 0.0, 0.0, 0.0, 1.0;

  for (int k = 0; k < N; ++k)
  {
    // 1) WORLD->TORSO from measurement
    const Eigen::Vector3d w_p_torso = qs.row(k).segment<3>(0).transpose();
    const Eigen::Vector4d w_q_torso_xyzw = qs.row(k).segment<4>(3).transpose();
    const pinocchio::SE3 world_M_torso = se3FromXYZQuat_xyzw(w_p_torso, w_q_torso_xyzw);

    // 2) pelvis->torso from kinematics with pelvis at identity
    q_fk.tail(model.nq - 7) = qs.row(k).tail(model.nq - 7).transpose();

    pinocchio::forwardKinematics(model, data, q_fk);
    pinocchio::updateFramePlacements(model, data);

    // With pelvis base identity, "o" frame equals pelvis frame
    const pinocchio::SE3 pelvis_M_torso = data.oMf[torso_fid];
    const pinocchio::SE3 torso_M_pelvis = pelvis_M_torso.inverse();

    // 3) WORLD->PELVIS = WORLD->TORSO * TORSO->PELVIS
    const pinocchio::SE3 world_M_pelvis = world_M_torso * torso_M_pelvis;

    // 4) Output as (x,y,z,qx,qy,qz,qw)
    pelvis_world.row(k).segment<3>(0) = world_M_pelvis.translation().transpose();
    pelvis_world.row(k).segment<4>(3) = quat_xyzw_FromSE3(world_M_pelvis).transpose();
  }

  return pelvis_world;
}

int main(int argc, char *argv[])
{
  // Data logging
  // ---------------------------------------------------------------------
  std::vector<std::string> files = {
      "data_g1_anitescu/high_kappa/f_log.csv",
      "data_g1_anitescu/high_kappa/mode_log.csv",
      "data_g1_anitescu/high_kappa/us_log.csv",
      "data_g1_anitescu/high_kappa/xs_log.csv",
      "data_g1_anitescu/high_kappa/us_results_fddp.csv",
      "data_g1_anitescu/high_kappa/xs_results_fddp.csv"};

  for (const auto &fpath : files)
  {
    if (std::filesystem::exists(fpath))
    {
      std::filesystem::remove(fpath);
    }
  }

  // Read parameters from XML file
  // ------------------------------------------------------------------
  char *path = "/home/jkang/third_party/PRIME/data_g1_anitescu/high_kappa/est_params_id.xml";
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
  {
    std::cerr << "Failed to open: " << path << "\n";
  }
  std::vector<char> buffer((std::istreambuf_iterator<char>(ifs)), {});
  buffer.push_back('\0');

  // Parse
  xml_document<> doc;
  doc.parse<0>(buffer.data());

  xml_node<> *root = doc.first_node("config");
  xml_node<> *xml_opt = root ? root->first_node("opt") : nullptr;

  double horizon = read_attr_double(xml_opt, "horizon", "value", 0.0);
  double interval = read_attr_double(xml_opt, "interval", "value", 0.0);
  double down_sample = read_attr_double(xml_opt, "down_sample", "value", 0.0);
  double max_iter = read_attr_double(xml_opt, "max_iter", "value", 0.0);
  double callbacks = read_attr_double(xml_opt, "callbacks", "value", 0.0);
  double start_idx = read_attr_double(xml_opt, "start_idx", "value", 0.0);
  double n_thread = read_attr_double(xml_opt, "n_thread", "value", 0.0);
  double alpha0 = read_attr_double(xml_opt, "alpha0", "value", 1.0);

  xml_node<> *xml_noise = root ? root->first_node("sim_noise") : nullptr;
  double q_noise = read_attr_double(xml_noise, "q_noise", "value", 0.0);
  double v_noise = read_attr_double(xml_noise, "v_noise", "value", 0.0);
  double u_noise = read_attr_double(xml_noise, "u_noise", "value", 0.0);

  // Model configuration
  // ---------------------------------------------------------------------
  std::string urdf_path = "/home/jkang/third_party/PRIME/data_g1_anitescu/unitree_description/urdf/g1/g1_29dof.urdf";
  std::string srdf_path = "/home/jkang/third_party/PRIME/data_g1_anitescu/unitree_description/srdf/main.srdf";

  pinocchio::Model model;
  pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model);
  pinocchio::srdf::loadReferenceConfigurations(model, srdf_path, false);
  std::vector<std::string> foot_frames = {
      "LR_FOOT_FL", "LR_FOOT_FR", "LR_FOOT_RL", "LR_FOOT_RR",
      "LL_FOOT_FL", "LL_FOOT_FR", "LL_FOOT_RL", "LL_FOOT_RR"};

  pinocchio::JointIndex j_joint = 16;
  std::vector<pinocchio::JointIndex> joints;
  joints.push_back(j_joint);

  Eigen::VectorXd pi_log = computeLogCholeskyFromLink(model, j_joint);
  // // pi_log[0] = pi_log[0] + 0.3;
  // std::cout << "Initial log–Cholesky parameters for joint " << j_joint << ":\n"
  //           << pi_log.transpose() << "\n";
  // std::cout << model.inertias[j_joint].toDynamicParameters().transpose() << "\n";
  Eigen::MatrixXd u0_init_noload = csvutil::readCSVtoEigen("data_g1_anitescu/high_kappa_nopayload/u0_results_fddp.csv");
  pi_log = pi_log + u0_init_noload.row(0).transpose();
  std::cout << "Initial log–Cholesky parameters for joint " << j_joint << ":\n"
            << pi_log.transpose() << "\n";
  crocoddyl::QuadrupedAnitescuIDProblem anitescu(model, joints, foot_frames, path);

  printPinocchioIndexOrder(model);
  pinocchio::Data data(model);

  const double timeStep = interval;
  Eigen::MatrixXd qs = csvutil::readCSVtoEigen("/home/jkang/third_party/PRIME/data_g1_anitescu/mocap_dance/p_sim.csv");
  Eigen::MatrixXd vs = csvutil::readCSVtoEigen("/home/jkang/third_party/PRIME/data_g1_anitescu/mocap_dance/v_sim.csv");
  Eigen::MatrixXd us = csvutil::readCSVtoEigen("/home/jkang/third_party/PRIME/data_g1_anitescu/mocap_dance/tau_sim.csv");
  const int T = qs.rows();
  const int nq = model.nq;
  const int nv = model.nv;

  Eigen::MatrixXd pelvis_world =
      computePelvisWorldFromTorsoBase_xyzw(model, "torso_link", qs.block(0, 1, T, nq));
  qs.block(0, 1, T, 7) = pelvis_world; // replace base with WORLD->PELVIS

  Eigen::MatrixXd xs_init = csvutil::readCSVtoEigen("data_g1_anitescu/xs_results_fddp.csv");
  Eigen::MatrixXd us_init = csvutil::readCSVtoEigen("data_g1_anitescu/us_results_fddp.csv");
  Eigen::MatrixXd u0_init = csvutil::readCSVtoEigen("data_g1_anitescu/u0_results_fddp.csv");

  // vs = lowpass_first_order_cutoff(vs, 0.005, 100.0); // dt=5ms, fc=20Hz
  // us = lowpass_first_order_cutoff(us, 0.005, 100.0); // dt=5ms, fc=20Hz

  Eigen::MatrixXd xs(T, nq + nv);
  xs.leftCols(nq) = qs.block(0, 1, T, nq);  // drop col 0
  xs.rightCols(nv) = vs.block(0, 1, T, nv); // drop col 0
  std::cout << "nq: " << model.nq << " nv: " << model.nv << std::endl;
  std::cout << "nq: " << nq << " nv: " << nv << std::endl;
  std::cout << "Read xs of size: " << xs.cols() << " x " << xs.cols() << std::endl;

  std::size_t ctrl_idx = 6;

  // -------------------------------------------------------------------
  Eigen::VectorXd x0 = anitescu.get_defaultState();
  x0.segment(0, model.nq) = xs.row(0 + start_idx).transpose().head(model.nq);
  x0.segment<4>(3).normalize();
  x0.segment(model.nq, 10 * joints.size()) = pi_log;
  x0.segment(model.nq + 10 * joints.size(), model.nv) = xs.row(0 + start_idx).transpose().tail(model.nv);

  double lowest_z = computeLowestFootHeight(model, data, x0.segment(0, model.nq));

  std::cout << "Lowest foot height (world z): " << lowest_z << std::endl;
  x0[2] = x0[2] - lowest_z;

  std::vector<Eigen::VectorXd> state_task;
  std::vector<Eigen::VectorXd> ctrl_task;

  Eigen::VectorXd xp = x0.segment(0, 3);
  for (std::size_t i = 0; i < horizon; i += down_sample)
  {
    Eigen::VectorXd xs_i = Eigen::VectorXd::Zero(model.nq + model.nv + 2 * 10 * joints.size());
    xs_i.segment(0, model.nq) = xs.row(i + start_idx).transpose().head(model.nq);
    xs_i.segment(model.nq, 10 * joints.size()) = pi_log;
    xs_i.segment(model.nq + 10 * joints.size(), model.nv) = xs.row(i + start_idx).transpose().tail(model.nv);
    lowest_z = computeLowestFootHeight(model, data, xs_i.segment(0, model.nq));
    xs_i[2] = xs_i[2] - lowest_z;
    xs_i.segment<4>(3).normalize();
    state_task.push_back(xs_i);
    csvutil::logVectorToCSV(xs_i, "data_g1_anitescu/high_kappa/xs_log.csv");
  }

  for (std::size_t i = 0; i + down_sample < horizon; i += down_sample)
  {
    const std::size_t row0 = start_idx + i;
    const std::size_t row1 = row0 + down_sample; // exclusive

    // Mean of controls over the block [row0, row1)
    Eigen::VectorXd ctrl_mean = Eigen::VectorXd::Zero(model.nv - 6);
    for (std::size_t r = row0; r < row1; ++r)
    {
      ctrl_mean.noalias() += us.row(r).segment(1, model.nv - 6).transpose();
    }
    ctrl_mean /= static_cast<double>(down_sample);

    // Add noise AFTER averaging (recommended)
    Eigen::VectorXd ctrl_noise = generateGaussianNoise(model.nv - 6, u_noise);

    Eigen::VectorXd ctrl_full_i = Eigen::VectorXd::Zero(model.nv);
    ctrl_full_i.segment(6, model.nv - 6) = ctrl_mean + ctrl_noise;
    ctrl_task.push_back(ctrl_full_i);
    csvutil::logVectorToCSV(ctrl_full_i, "data_g1_anitescu/high_kappa/us_log.csv");
  }

  std::size_t ddp_knotes = state_task.size();

  std::vector<Eigen::VectorXd> state_init;
  std::vector<Eigen::VectorXd> ctrl_init;

  // state_task.push_back(x0);
  state_init.push_back(x0);
  ctrl_init.push_back(Eigen::VectorXd::Zero(10));
  // ctrl_init.push_back(u0_init.row(0).transpose());

  std::cout << "state size: " << ddp_knotes << std::endl;
  std::cout << "ctrl size: " << ctrl_task.size() << std::endl;
  for (std::size_t i = 0; i < ddp_knotes; i += 1)
  {

    Eigen::VectorXd xs_i_init = xs_init.row(i + 1).transpose();
    state_init.push_back(xs_i_init);
  }
  for (std::size_t i = 0; i < ddp_knotes - 1; i += 1)
  {
    Eigen::VectorXd ctrl_i_init = us_init.row(i).transpose();
    ctrl_init.push_back(ctrl_i_init);
  }

  // for (std::size_t i = 0; i < horizon - down_sample; i += down_sample)
  // {
  //   Eigen::VectorXd ctrl_i = us.row(i + start_idx).segment(0, 18).transpose(); // Add some noise
  //   Eigen::VectorXd ctrl_i_18 = Eigen::VectorXd::Zero(18);
  //   ctrl_i_18.segment<12>(6) = ctrl_i.segment<12>(ctrl_idx);
  //   ctrl_task.push_back(ctrl_i_18);
  //   // ctrl_task.push_back(Eigen::VectorXd::Zero(18));

  //   ctrl_init.push_back(ctrl_i_18);
  //   // ctrl_init.push_back(Eigen::VectorXd::Zero(18));
  //   csvutil::logVectorToCSV(ctrl_i_18, "data_g1_anitescu/us_log.csv");
  // }

  // DDP Solver
  boost::shared_ptr<crocoddyl::ShootingProblem> problem =
      anitescu.createEstimationProblem(x0, down_sample * timeStep, ddp_knotes, state_task, ctrl_task);
  problem->set_nthreads(n_thread);

  // crocoddyl::SolverDDP solver(problem);
  crocoddyl::SolverFDDP solver(problem);

  int dim_aplha = 11;

  // alpha0, alpha0/2, alpha0/4, ...
  std::vector<double> alphas;
  alphas.reserve(dim_aplha);
  for (int i = 0; i < dim_aplha; ++i)
  {
    alphas.push_back(std::ldexp(alpha0, -i)); // alpha0 * 2^{-i}
  }

  solver.set_alphas(alphas);

  if (callbacks)
  {
    std::vector<boost::shared_ptr<crocoddyl::CallbackAbstract>> cbs;
    cbs.push_back(boost::make_shared<crocoddyl::CallbackVerbose>());
    solver.setCallbacks(cbs);
  }

  // Initial State
  const std::size_t N = solver.get_problem()->get_T();
  // std::vector<Eigen::VectorXd> xs_init = state_task; // Use the state task as initial state
  // std::vector<Eigen::VectorXd> us_init = ctrl_task;  // Use the control task as initial control

  // std::vector<Eigen::VectorXd> xs_init(N + 1, x0);
  // std::vector<Eigen::VectorXd> us_init(N, Eigen::VectorXd::Zero(ctrl_task[0].size()));

  std::cout << "NQ: "
            << solver.get_problem()->get_terminalModel()->get_state()->get_nq()
            << std::endl;
  std::cout << "Number of nodes: " << N << std::endl;

  // ---------------------------------------------------------------------
  // Solving the optimal control problem
  Eigen::ArrayXd duration(1);

  crocoddyl::Timer timer;
  solver.solve(state_init, ctrl_init, max_iter, false, 0.1);
  duration[0] = timer.get_duration();
  std::cout << "Duration: " << duration[0] / 1000 << " seconds" << std::endl;

  // ---------------------------------------------------------------------
  // Log the solution
  const std::vector<Eigen::VectorXd> &xs_sol = solver.get_xs(); // States
  const std::vector<Eigen::VectorXd> &us_sol = solver.get_us(); // Controls
  const std::vector<Eigen::VectorXd> &Vx_sol = solver.get_Vx();
  const std::vector<Eigen::MatrixXd> &Vxx_sol = solver.get_Vxx();
  const std::vector<Eigen::VectorXd> &Qu_sol = solver.get_Qu();

  // Eigen::VectorXd dp_0 = -Vxx_sol[0].block(18, 18, 10, 10).inverse() * Vx_sol[0].segment(18, 10);

  // std::cout << "V_x" << std::endl
  //           << Vx_sol[0].transpose() << std::endl;
  // std::cout << "V_xx" << std::endl
  //           << Vxx_sol[0].block(18, 18, 10, 10) << std::endl;
  // std::cout << "dp_0" << std::endl
  //           << dp_0.transpose() << std::endl;
  // std::cout << "Qu_0" << std::endl
  //           << Qu_sol[0].transpose() << std::endl;

  // std::cout << "dx_0" << std::endl
  //           << xs_sol[1].transpose() << std::endl;
  std::cout << "u_0" << std::endl
            << us_sol[0].transpose() << std::endl;
  std::cout << "init" << std::endl
            << pi_log.transpose() << std::endl;
  std::cout << "sol" << std::endl
            << xs_sol[1].transpose().segment(model.nq, 10) << std::endl;
  // pi_log[0] = pi_log[0] - 0.3;
  std::cout << "true" << std::endl
            << pi_log.transpose() << std::endl;

  Eigen::VectorXd eta1 = xs_sol[1].segment(model.nq, 10); // your "dx_0" printed segment
  Eigen::VectorXd eta2 = pi_log;                          // your "log"
  // eta2[0] = eta2[0] - 0.2;
  // 1) Log-Cholesky params -> dynamic parameters (10) that represent pseudo-inertia
  pinocchio::LogCholeskyParametersTpl<double> lc1(eta1);
  pinocchio::LogCholeskyParametersTpl<double> lc2(eta2);

  Eigen::Matrix<double, 10, 1> pi1 = lc1.toDynamicParameters();
  Eigen::Matrix<double, 10, 1> pi2 = lc2.toDynamicParameters();

  // 2) Dynamic parameters -> Inertia -> PseudoInertia
  pinocchio::Inertia I1 = pinocchio::Inertia::FromDynamicParameters(pi1);
  pinocchio::Inertia I2 = pinocchio::Inertia::FromDynamicParameters(pi2);

  pinocchio::PseudoInertia J1 = I1.toPseudoInertia();
  pinocchio::PseudoInertia J2 = I2.toPseudoInertia();

  // 3) Print 4x4 pseudo-inertia matrices
  std::cout << "Pseudo-inertia from xs_sol:\n"
            << J1.toMatrix() << "\n\n";
  std::cout << "Dynamic parameters from xs_sol:\n"
            << I1.toDynamicParameters().transpose() << "\n\n";

  std::cout << "Pseudo-inertia from pi_log:\n"
            << J2.toMatrix() << "\n";
  std::cout << "Dynamic parameters from pi_log:\n"
            << I2.toDynamicParameters().transpose() << "\n\n";

  Eigen::MatrixXd xs_results = Eigen::MatrixXd::Zero(xs_sol.size(), xs_sol[0].size());
  Eigen::MatrixXd us_results = Eigen::MatrixXd::Zero(us_sol.size() - 1, us_sol[1].size());
  for (std::size_t i = 0; i < xs_sol.size(); ++i)
  {
    xs_results.row(i) = xs_sol[i].transpose();
  }
  for (std::size_t i = 1; i < us_sol.size() - 1; ++i)
  {
    int nu = us_sol[i].size();
    us_results.row(i).segment(0, nu) = us_sol[i].transpose();
  }
  csvutil::saveEigenToCSV("data_g1_anitescu/high_kappa/xs_results_fddp.csv", xs_results);
  csvutil::saveEigenToCSV("data_g1_anitescu/high_kappa/us_results_fddp.csv", us_results);

  const int nu0 = us_sol[0].size();
  Eigen::MatrixXd u0_mat(1, nu0);
  u0_mat.row(0) = us_sol[0].transpose();
  csvutil::saveEigenToCSV("data_g1_anitescu/high_kappa/u0_results_fddp.csv", u0_mat);

  // Forward Rollout
  // ---------------------------------------------------------------------
  std::string forcepath = "data_g1_anitescu/f_rollout.csv";
  if (std::filesystem::exists(forcepath))
  {
    std::filesystem::remove(forcepath);
  }

  std::vector<Eigen::VectorXd> xs_rollout(N + 1);
  std::vector<Eigen::VectorXd> us_rollout = us_sol;
  problem->rollout(us_rollout, xs_rollout);
  Eigen::MatrixXd xs_roll = Eigen::MatrixXd::Zero(xs_rollout.size(), xs_rollout[0].size());
  for (std::size_t i = 0; i < xs_rollout.size(); ++i)
  {
    xs_roll.row(i) = xs_rollout[i].transpose(); // Convert VectorXd to RowVectorXd
  }
  csvutil::saveEigenToCSV("data_g1_anitescu/high_kappa/xs_rollout.csv", xs_roll);
}
