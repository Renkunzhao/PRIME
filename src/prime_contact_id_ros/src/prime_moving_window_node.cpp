///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// ROS2 bridge for local PRIME moving-window inertial identification.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include <Eigen/Dense>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"
#include "crocoddyl/contact_id/config/contact-id-config.hpp"
#include "crocoddyl/contact_id/problem/contact-anitescu-id.hpp"

#include "contact_id_model.hpp"
#include "contact_id_outputs.hpp"
#include "contact_id_preprocess.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct PrimeSample {
  double stamp_ns;
  Eigen::VectorXd q;
  Eigen::VectorXd v;
  Eigen::VectorXd tau;
};

struct ExtractedMarginal {
  crocoddyl::MarginalizedArrivalPrior prior;
  Eigen::MatrixXd raw_information;
  Eigen::VectorXd raw_gradient;
  double min_eig;
  double max_eig;
  std::string status;
};

bool compiled_with_multithreading() {
#ifdef CROCODDYL_WITH_MULTITHREADING
  return true;
#else
  return false;
#endif
}

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

std::string join_path(const std::string& lhs, const std::string& rhs) {
  return crocoddyl::contact_id_xml::join_path(lhs, rhs);
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

Eigen::VectorXd flatten_parameter_logs(
    const std::vector<Eigen::VectorXd>& parameter_logs) {
  Eigen::VectorXd theta = Eigen::VectorXd::Zero(
      static_cast<Eigen::Index>(10 * parameter_logs.size()));
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

void apply_shifted_warm_start(
    const pinocchio::Model& model, const Eigen::VectorXd& theta,
    std::size_t stride_knots, const std::vector<Eigen::VectorXd>& previous_xs,
    const std::vector<Eigen::VectorXd>& previous_us,
    contact_id_experiment::PreparedContactIDData& data) {
  if (previous_xs.empty() || previous_us.empty() || stride_knots == 0 ||
      stride_knots >= previous_xs.size()) {
    overwrite_theta_sequence(data.state_init, model.nq, theta);
    return;
  }

  const std::size_t state_limit =
      std::min(data.state_init.size(), previous_xs.size() - stride_knots);
  for (std::size_t i = 1; i < state_limit; ++i) {
    data.state_init[i] = previous_xs[stride_knots + i];
  }

  if (stride_knots + 1 < previous_us.size()) {
    const std::size_t control_limit =
        std::min(data.ctrl_init.size(), previous_us.size() - stride_knots);
    for (std::size_t i = 1; i < control_limit; ++i) {
      data.ctrl_init[i] = previous_us[stride_knots + i];
    }
  }
  overwrite_theta_sequence(data.state_init, model.nq, theta);
}

crocoddyl::ContactIDWeights without_arrival_regularization(
    const crocoddyl::ContactIDWeights& weights) {
  crocoddyl::ContactIDWeights out = weights;
  out.arrival_alpha = 0.;
  out.arrival_com = 0.;
  out.arrival_diag = 0.;
  return out;
}

crocoddyl::ContactIDOutputConfig quiet_outputs(
    const crocoddyl::ContactIDOutputConfig& outputs) {
  crocoddyl::ContactIDOutputConfig out = outputs;
  out.force_log.clear();
  out.log_initial_guess = false;
  out.rollout = false;
  out.save_force_rollout = false;
  return out;
}

crocoddyl::MarginalizedArrivalPrior initial_arrival_prior(
    const pinocchio::Model& model,
    const std::vector<pinocchio::JointIndex>& joints,
    const crocoddyl::ContactIDWeights& weights, const Eigen::VectorXd& theta) {
  const Eigen::Index np = theta.size();
  Eigen::MatrixXd information = Eigen::MatrixXd::Zero(np, np);
  for (std::size_t j = 0; j < joints.size(); ++j) {
    Eigen::VectorXd defect_weights = Eigen::VectorXd::Zero(10);
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
    information.diagonal().segment(static_cast<Eigen::Index>(10 * j), 10) =
        10. * defect_weights.array() * s_link.array().square();
  }
  return crocoddyl::MarginalizedArrivalPrior(theta, information);
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
  if (compiled_with_multithreading()) {
    shooting_problem->set_nthreads(static_cast<int>(solver_cfg.n_thread));
  }
  return shooting_problem;
}

contact_id_experiment::PreparedContactIDData dropped_chunk_data(
    const contact_id_experiment::PreparedContactIDData& long_data,
    const std::vector<Eigen::VectorXd>& xs_solution,
    std::size_t dropped_knots) {
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
  if (dropped_knots == 0 || long_data.state_task.size() < dropped_knots ||
      long_data.ctrl_task.size() + 1 < dropped_knots ||
      xs_solution.size() < dropped_knots + 1 ||
      us_solution.size() < dropped_knots) {
    throw std::runtime_error("Dropped horizon is larger than solved window.");
  }

  contact_id_experiment::PreparedContactIDData drop_data =
      dropped_chunk_data(long_data, xs_solution, dropped_knots);
  crocoddyl::ContactIDSolverConfig drop_solver = window_cfg.solver;
  drop_solver.horizon = dropped_knots * window_cfg.solver.down_sample;
  drop_solver.n_thread = 1;

  const crocoddyl::ContactIDWeights drop_weights =
      without_arrival_regularization(window_cfg.weights);
  const crocoddyl::ContactIDOutputConfig drop_outputs =
      quiet_outputs(window_cfg.outputs);

  boost::shared_ptr<crocoddyl::ShootingProblem> drop_problem =
      create_problem(model, joints, contact_frames, drop_weights, drop_solver,
                     drop_outputs, crocoddyl::MarginalizedArrivalPrior(),
                     drop_data);
  std::vector<Eigen::VectorXd> us_drop(us_solution.begin(),
                                       us_solution.begin() + dropped_knots);
  std::vector<Eigen::VectorXd> xs_feasible(drop_problem->get_T() + 1);
  drop_problem->rollout(us_drop, xs_feasible);

  crocoddyl::SolverFDDP extractor(drop_problem);
  extractor.set_alphas(make_solver_alphas(1.));
  extractor.setCandidate(xs_feasible, us_drop, true);
  extractor.calcDiff();
  extractor.backwardPass();

  double min_eig = 0.;
  double max_eig = 0.;
  const Eigen::MatrixXd information = clamp_information(
      extractor.get_Quu()[0], information_floor, information_ceiling,
      min_eig, max_eig);
  const Eigen::VectorXd gradient = extractor.get_Qu()[0];
  const Eigen::VectorXd theta_current =
      xs_feasible[1].segment(model.nq, us_drop[0].size());

  Eigen::LDLT<Eigen::MatrixXd> ldlt(information);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("Failed to solve dropped information system.");
  }

  ExtractedMarginal out;
  out.prior = crocoddyl::MarginalizedArrivalPrior(
      theta_current - ldlt.solve(gradient), information);
  out.raw_information = extractor.get_Quu()[0];
  out.raw_gradient = gradient;
  out.min_eig = min_eig;
  out.max_eig = max_eig;
  out.status = "dropped_horizon_backward";
  return out;
}

ExtractedMarginal accumulate_parameter_information(
    const crocoddyl::MarginalizedArrivalPrior& carried_prior,
    const ExtractedMarginal& dropped_information,
    double information_floor, double information_ceiling,
    double information_forgetting_factor) {
  const Eigen::Index np = dropped_information.prior.mean.size();
  Eigen::MatrixXd information = Eigen::MatrixXd::Zero(np, np);
  Eigen::VectorXd eta = Eigen::VectorXd::Zero(np);
  if (carried_prior.enabled) {
    information.noalias() +=
        information_forgetting_factor * carried_prior.information;
    eta.noalias() += information_forgetting_factor *
                     carried_prior.information * carried_prior.mean;
  }
  information.noalias() += dropped_information.prior.information;
  eta.noalias() +=
      dropped_information.prior.information * dropped_information.prior.mean;

  double min_eig = 0.;
  double max_eig = 0.;
  const Eigen::MatrixXd clamped = clamp_information(
      information, information_floor, information_ceiling, min_eig, max_eig);
  Eigen::LDLT<Eigen::MatrixXd> ldlt(clamped);
  if (ldlt.info() != Eigen::Success) {
    throw std::runtime_error("Failed to solve accumulated information system.");
  }

  ExtractedMarginal out;
  out.prior = crocoddyl::MarginalizedArrivalPrior(ldlt.solve(eta), clamped);
  out.raw_information = information;
  out.raw_gradient = Eigen::VectorXd::Zero(np);
  out.min_eig = min_eig;
  out.max_eig = max_eig;
  out.status = dropped_information.status;
  return out;
}

std::vector<std::string> actuated_joint_names(const pinocchio::Model& model) {
  std::vector<std::string> names;
  for (pinocchio::JointIndex j = 1; j < model.njoints; ++j) {
    if (model.nqs[j] == 1 && model.idx_qs[j] >= 7) {
      names.push_back(model.names[j]);
    }
  }
  return names;
}

std::string to_string(double value) {
  std::ostringstream os;
  os << std::setprecision(12) << value;
  return os.str();
}

void add_kv(diagnostic_msgs::msg::DiagnosticStatus& status,
            const std::string& key, const std::string& value) {
  diagnostic_msgs::msg::KeyValue kv;
  kv.key = key;
  kv.value = value;
  status.values.push_back(kv);
}

}  // namespace

class PrimeMovingWindowNode : public rclcpp::Node {
 public:
  PrimeMovingWindowNode()
      : Node("prime_moving_window_node"),
        model_(),
        has_odom_(false),
        solving_(false),
        missed_cycles_(0),
        next_window_(0),
        online_start_idx_(0),
        save_raw_logs_(false),
        clear_results_on_start_(false) {
    const std::string config_path =
        declare_parameter<std::string>("config_path", "");
    const std::string odom_topic =
        declare_parameter<std::string>("base_odom_topic", "/odom");
    const std::string joint_topic =
        declare_parameter<std::string>("joint_state_topic", "/joint_states");
    const std::string torque_topic =
        declare_parameter<std::string>("torque_topic", "");
    online_start_idx_ = static_cast<std::size_t>(
        declare_parameter<int>("online_start_idx", 0));
    save_raw_logs_ = declare_parameter<bool>("save_raw_logs", false);
    clear_results_on_start_ =
        declare_parameter<bool>("clear_results_on_start", false);

    if (config_path.empty()) {
      throw std::runtime_error("Parameter config_path is required.");
    }

    cfg_ = crocoddyl::contact_id_xml::load_config(config_path);
    const bool solver_verbose =
        declare_parameter<bool>("solver_verbose", cfg_.solver.callbacks);
    cfg_.solver.callbacks = solver_verbose;
    contact_id_experiment::validate_required(cfg_);
    cfg_.data.q_has_time_column = true;
    cfg_.data.v_has_time_column = true;
    cfg_.data.u_has_time_column = true;
    cfg_.data.joint_order.clear();
    cfg_.data.torso_base_frame.clear();

    model_ = contact_id_experiment::load_floating_base_model(cfg_.robot);
    joints_ = contact_id_experiment::resolve_identified_joints(model_, cfg_);
    joint_names_ = actuated_joint_names(model_);
    if (joint_names_.size() != static_cast<std::size_t>(model_.nq - 7)) {
      throw std::runtime_error("Could not infer all actuated joint names.");
    }

    logs_template_.parameter_logs =
        contact_id_experiment::compute_parameter_logs(
            model_, joints_, cfg_.parameter_offsets);

    if (clear_results_on_start_) {
      contact_id_experiment::clear_directory_contents(cfg_.outputs.directory);
    }
    if (save_raw_logs_) {
      contact_id_experiment::ensure_dir(cfg_.outputs.directory);
      contact_id_experiment::truncate_file(join_path(cfg_.outputs.directory,
                                                     "ros_q_log.csv"));
      contact_id_experiment::truncate_file(join_path(cfg_.outputs.directory,
                                                     "ros_v_log.csv"));
      contact_id_experiment::truncate_file(join_path(cfg_.outputs.directory,
                                                     "ros_tau_log.csv"));
    }

    theta_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        "~/theta_log_cholesky", 10);
    inertia_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(
        "~/inertia_dynamic_params", 10);
    diag_pub_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
        "~/solver_stats", 10);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic, 50,
        std::bind(&PrimeMovingWindowNode::odomCallback, this,
                  std::placeholders::_1));
    joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
        joint_topic, 200,
        std::bind(&PrimeMovingWindowNode::jointCallback, this,
                  std::placeholders::_1));
    if (!torque_topic.empty()) {
      torque_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
          torque_topic, 50,
          std::bind(&PrimeMovingWindowNode::torqueCallback, this,
                    std::placeholders::_1));
    }

    const double period_s = static_cast<double>(cfg_.moving_horizon.stride_knots) *
                            static_cast<double>(cfg_.solver.down_sample) *
                            cfg_.solver.interval;
    if (!(period_s > 0.)) {
      throw std::runtime_error("Computed non-positive timer period.");
    }
    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(period_s)),
        std::bind(&PrimeMovingWindowNode::timerCallback, this));

    RCLCPP_INFO(get_logger(),
                "PRIME moving-window node ready. timer_period=%.6f s, horizon=%zu samples, stride=%zu samples, compiled_multithreading=%d, solver_verbose=%d",
                period_s, cfg_.solver.horizon,
                cfg_.moving_horizon.stride_knots * cfg_.solver.down_sample,
                compiled_with_multithreading(), cfg_.solver.callbacks);
  }

 private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_odom_ = *msg;
    has_odom_ = true;
  }

  void torqueCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_torque_ = msg->data;
  }

  void jointCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_odom_) {
      return;
    }

    PrimeSample sample;
    sample.stamp_ns = static_cast<double>(msg->header.stamp.sec) * 1e9 +
                      static_cast<double>(msg->header.stamp.nanosec);
    if (sample.stamp_ns == 0.) {
      sample.stamp_ns = static_cast<double>(now().nanoseconds());
    }
    sample.q = Eigen::VectorXd::Zero(model_.nq);
    sample.v = Eigen::VectorXd::Zero(model_.nv);
    sample.tau = Eigen::VectorXd::Zero(model_.nv - 6);

    sample.q[0] = latest_odom_.pose.pose.position.x;
    sample.q[1] = latest_odom_.pose.pose.position.y;
    sample.q[2] = latest_odom_.pose.pose.position.z;
    sample.q[3] = latest_odom_.pose.pose.orientation.x;
    sample.q[4] = latest_odom_.pose.pose.orientation.y;
    sample.q[5] = latest_odom_.pose.pose.orientation.z;
    sample.q[6] = latest_odom_.pose.pose.orientation.w;
    sample.q.segment<4>(3).normalize();

    sample.v[0] = latest_odom_.twist.twist.linear.x;
    sample.v[1] = latest_odom_.twist.twist.linear.y;
    sample.v[2] = latest_odom_.twist.twist.linear.z;
    sample.v[3] = latest_odom_.twist.twist.angular.x;
    sample.v[4] = latest_odom_.twist.twist.angular.y;
    sample.v[5] = latest_odom_.twist.twist.angular.z;

    for (std::size_t i = 0; i < joint_names_.size(); ++i) {
      const std::vector<std::string>::const_iterator it =
          std::find(msg->name.begin(), msg->name.end(), joint_names_[i]);
      if (it == msg->name.end()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "JointState missing required joint '%s'.",
                             joint_names_[i].c_str());
        return;
      }
      const std::size_t raw = static_cast<std::size_t>(it - msg->name.begin());
      if (raw >= msg->position.size() || raw >= msg->velocity.size()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                             "JointState position/velocity arrays are incomplete.");
        return;
      }
      sample.q[7 + static_cast<Eigen::Index>(i)] = msg->position[raw];
      sample.v[6 + static_cast<Eigen::Index>(i)] = msg->velocity[raw];
      if (raw < msg->effort.size()) {
        sample.tau[static_cast<Eigen::Index>(i)] = msg->effort[raw];
      } else if (latest_torque_.size() == joint_names_.size()) {
        sample.tau[static_cast<Eigen::Index>(i)] = latest_torque_[i];
      }
    }

    samples_.push_back(sample);
    if (save_raw_logs_) {
      logRawSample(sample);
    }
  }

  void timerCallback() {
    contact_id_experiment::LoadedContactIDLogs logs;
    std::size_t window_index = 0;
    std::size_t global_start = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (solving_) {
        ++missed_cycles_;
        publishDiagnosticOnly("solve_busy");
        return;
      }
      if (cfg_.moving_horizon.enabled && next_window_ >= cfg_.moving_horizon.max_windows) {
        return;
      }
      const std::size_t stride_samples =
          cfg_.moving_horizon.stride_knots * cfg_.solver.down_sample;
      global_start = online_start_idx_ + next_window_ * stride_samples;
      if (samples_.size() < global_start + cfg_.solver.horizon) {
        publishDiagnosticOnly("waiting_for_full_window");
        return;
      }
      logs = makeWindowLogs(global_start);
      window_index = next_window_;
      solving_ = true;
      ++next_window_;
    }

    std::thread(&PrimeMovingWindowNode::solveWindow, this, logs, window_index,
                global_start)
        .detach();
  }

  contact_id_experiment::LoadedContactIDLogs makeWindowLogs(
      std::size_t global_start) const {
    contact_id_experiment::LoadedContactIDLogs logs = logs_template_;
    const std::size_t rows = cfg_.solver.horizon;
    logs.q_processed = Eigen::MatrixXd::Zero(rows, model_.nq + 1);
    logs.v_processed = Eigen::MatrixXd::Zero(rows, model_.nv + 1);
    logs.u_processed = Eigen::MatrixXd::Zero(rows, model_.nv - 6 + 1);
    for (std::size_t r = 0; r < rows; ++r) {
      const PrimeSample& sample = samples_[global_start + r];
      logs.q_processed(static_cast<Eigen::Index>(r), 0) = sample.stamp_ns;
      logs.v_processed(static_cast<Eigen::Index>(r), 0) = sample.stamp_ns;
      logs.u_processed(static_cast<Eigen::Index>(r), 0) = sample.stamp_ns;
      logs.q_processed.row(static_cast<Eigen::Index>(r)).segment(1, model_.nq) =
          sample.q.transpose();
      logs.v_processed.row(static_cast<Eigen::Index>(r)).segment(1, model_.nv) =
          sample.v.transpose();
      logs.u_processed.row(static_cast<Eigen::Index>(r)).segment(1, model_.nv - 6) =
          sample.tau.transpose();
    }
    return logs;
  }

  void solveWindow(contact_id_experiment::LoadedContactIDLogs logs,
                   std::size_t window_index, std::size_t global_start) {
    const Clock::time_point start = Clock::now();
    double cost = std::numeric_limits<double>::quiet_NaN();
    double feasibility = std::numeric_limits<double>::quiet_NaN();
    ExtractedMarginal marginal;
    try {
      RCLCPP_INFO(get_logger(), "Solving PRIME moving-window %zu, buffered_start=%zu, dense_prior=%d",
                  window_index, global_start, carried_prior_.enabled);

      crocoddyl::ContactIDXMLConfig window_cfg = cfg_;
      window_cfg.solver.start_idx = 0;

      contact_id_experiment::PreparedContactIDData data =
          contact_id_experiment::prepare_contact_id_data(
              model_, joints_, cfg_.contact_frames, window_cfg, logs);

      crocoddyl::MarginalizedArrivalPrior arrival_prior;
      Eigen::VectorXd theta;
      std::vector<Eigen::VectorXd> previous_xs;
      std::vector<Eigen::VectorXd> previous_us;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (window_index == 0 || !carried_prior_.enabled) {
          theta = flatten_parameter_logs(data.parameter_logs);
          carried_prior_ = initial_arrival_prior(model_, joints_, cfg_.weights, theta);
          carried_theta_ = theta;
        } else {
          theta = carried_theta_;
        }
        arrival_prior = carried_prior_;
        previous_xs = previous_xs_;
        previous_us = previous_us_;
      }

      apply_prior_mean_to_data(model_, theta, data);
      if (window_index > 0) {
        apply_shifted_warm_start(model_, theta, cfg_.moving_horizon.stride_knots,
                                 previous_xs, previous_us, data);
      }

      crocoddyl::ContactIDOutputConfig outputs = cfg_.outputs;
      outputs.force_log.clear();
      boost::shared_ptr<crocoddyl::ShootingProblem> problem = create_problem(
          model_, joints_, cfg_.contact_frames, cfg_.weights, window_cfg.solver,
          outputs, arrival_prior, data);

      crocoddyl::SolverFDDP solver(problem);
      solver.set_alphas(make_solver_alphas(cfg_.solver.alpha0));
      configure_callbacks(solver, cfg_.solver.callbacks);
      solver.solve(data.state_init, data.ctrl_init, cfg_.solver.max_iter, false,
                   0.1);
      cost = solver.get_cost();
      feasibility = solver.computeDynamicFeasibility();

      const ExtractedMarginal dropped = extract_dropped_chunk_information(
          model_, joints_, cfg_.contact_frames, window_cfg, data,
          solver.get_xs(), solver.get_us(), cfg_.moving_horizon.stride_knots,
          cfg_.moving_horizon.information_floor,
          cfg_.moving_horizon.information_ceiling);
      const double forgetting = cfg_.moving_horizon.information_forgetting_enabled
                                    ? cfg_.moving_horizon.information_forgetting_factor
                                    : 1.;
      marginal = accumulate_parameter_information(
          arrival_prior, dropped, cfg_.moving_horizon.information_floor,
          cfg_.moving_horizon.information_ceiling, forgetting);

      {
        std::lock_guard<std::mutex> lock(mutex_);
        carried_prior_ = marginal.prior;
        carried_theta_ = marginal.prior.mean;
        previous_xs_ = solver.get_xs();
        previous_us_ = solver.get_us();
      }

      const double total_ms = elapsedMs(start, Clock::now());
      RCLCPP_INFO(get_logger(),
                  "Finished PRIME moving-window %zu: cost=%.6e feasibility=%.6e total_ms=%.3f min_info=%.6e max_info=%.6e",
                  window_index, cost, feasibility, total_ms, marginal.min_eig,
                  marginal.max_eig);
      publishResult(window_index, global_start, cost, feasibility, marginal,
                    total_ms, "ok");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "PRIME solve failed on window %zu: %s",
                   window_index, e.what());
      publishResult(window_index, global_start, cost, feasibility, marginal,
                    elapsedMs(start, Clock::now()), e.what());
    }

    std::lock_guard<std::mutex> lock(mutex_);
    solving_ = false;
  }

  double elapsedMs(const Clock::time_point& a, const Clock::time_point& b) const {
    return std::chrono::duration<double, std::milli>(b - a).count();
  }

  void publishResult(std::size_t window_index, std::size_t global_start,
                     double cost, double feasibility,
                     const ExtractedMarginal& marginal, double total_ms,
                     const std::string& status_text) {
    if (marginal.prior.enabled) {
      std_msgs::msg::Float64MultiArray theta_msg;
      theta_msg.data.resize(static_cast<std::size_t>(marginal.prior.mean.size()));
      for (Eigen::Index i = 0; i < marginal.prior.mean.size(); ++i) {
        theta_msg.data[static_cast<std::size_t>(i)] = marginal.prior.mean[i];
      }
      theta_pub_->publish(theta_msg);

      std_msgs::msg::Float64MultiArray inertia_msg;
      for (Eigen::Index i = 0; i < marginal.prior.mean.size(); i += 10) {
        const pinocchio::Inertia inertia =
            contact_id_experiment::inertia_from_log_cholesky(
                marginal.prior.mean.segment(i, 10));
        const Eigen::Matrix<double, 10, 1> params =
            inertia.toDynamicParameters();
        for (int k = 0; k < 10; ++k) {
          inertia_msg.data.push_back(params[k]);
        }
      }
      inertia_pub_->publish(inertia_msg);
    }

    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "prime_moving_window_node";
    status.hardware_id = "prime_contact_id";
    status.level = status_text == "ok" ?
                       diagnostic_msgs::msg::DiagnosticStatus::OK :
                       diagnostic_msgs::msg::DiagnosticStatus::ERROR;
    status.message = status_text;
    add_kv(status, "window", std::to_string(window_index));
    add_kv(status, "sample_start", std::to_string(global_start));
    add_kv(status, "cost", to_string(cost));
    add_kv(status, "feasibility", to_string(feasibility));
    add_kv(status, "total_ms", to_string(total_ms));
    add_kv(status, "min_info_eig", to_string(marginal.min_eig));
    add_kv(status, "max_info_eig", to_string(marginal.max_eig));
    add_kv(status, "missed_cycles", std::to_string(missed_cycles_.load()));
    array.status.push_back(status);
    diag_pub_->publish(array);
  }

  void publishDiagnosticOnly(const std::string& message) {
    diagnostic_msgs::msg::DiagnosticArray array;
    array.header.stamp = now();
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.name = "prime_moving_window_node";
    status.hardware_id = "prime_contact_id";
    status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
    status.message = message;
    add_kv(status, "buffered_samples", std::to_string(samples_.size()));
    add_kv(status, "next_window", std::to_string(next_window_));
    add_kv(status, "missed_cycles", std::to_string(missed_cycles_.load()));
    array.status.push_back(status);
    diag_pub_->publish(array);
  }

  void logRawSample(const PrimeSample& sample) const {
    Eigen::VectorXd q_row(model_.nq + 1);
    Eigen::VectorXd v_row(model_.nv + 1);
    Eigen::VectorXd u_row(model_.nv - 6 + 1);
    q_row[0] = sample.stamp_ns;
    v_row[0] = sample.stamp_ns;
    u_row[0] = sample.stamp_ns;
    q_row.tail(model_.nq) = sample.q;
    v_row.tail(model_.nv) = sample.v;
    u_row.tail(model_.nv - 6) = sample.tau;
    csvutil::logVectorToCSV(q_row, join_path(cfg_.outputs.directory, "ros_q_log.csv"));
    csvutil::logVectorToCSV(v_row, join_path(cfg_.outputs.directory, "ros_v_log.csv"));
    csvutil::logVectorToCSV(u_row, join_path(cfg_.outputs.directory, "ros_tau_log.csv"));
  }

  crocoddyl::ContactIDXMLConfig cfg_;
  pinocchio::Model model_;
  std::vector<pinocchio::JointIndex> joints_;
  std::vector<std::string> joint_names_;
  contact_id_experiment::LoadedContactIDLogs logs_template_;

  std::mutex mutex_;
  nav_msgs::msg::Odometry latest_odom_;
  bool has_odom_;
  std::vector<double> latest_torque_;
  std::deque<PrimeSample> samples_;
  bool solving_;
  std::atomic<std::size_t> missed_cycles_;
  std::size_t next_window_;
  std::size_t online_start_idx_;
  bool save_raw_logs_;
  bool clear_results_on_start_;
  crocoddyl::MarginalizedArrivalPrior carried_prior_;
  Eigen::VectorXd carried_theta_;
  std::vector<Eigen::VectorXd> previous_xs_;
  std::vector<Eigen::VectorXd> previous_us_;

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr torque_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr theta_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr inertia_pub_;
  rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diag_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<PrimeMovingWindowNode>());
  } catch (const std::exception& e) {
    std::cerr << "prime_moving_window_node: " << e.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
