///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// CSV replay helper for the PRIME ROS2 moving-window node.
///////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "crocoddyl/contact_id/config/contact-id-config.hpp"
#include "contact_id_model.hpp"
#include "contact_id_preprocess.hpp"

namespace {

std::vector<std::string> actuated_joint_names(const pinocchio::Model& model) {
  std::vector<std::string> names;
  for (pinocchio::JointIndex j = 1; j < model.njoints; ++j) {
    if (model.nqs[j] == 1 && model.idx_qs[j] >= 7) {
      names.push_back(model.names[j]);
    }
  }
  return names;
}

}  // namespace

class PrimeCsvReplayNode : public rclcpp::Node {
 public:
  PrimeCsvReplayNode() : Node("prime_csv_replay_node"), row_(0), end_row_(0) {
    const std::string config_path =
        declare_parameter<std::string>("config_path", "");
    const int start_idx_param = declare_parameter<int>("start_idx", -1);
    const int max_samples_param = declare_parameter<int>("max_samples", 0);
    const int stride_param = declare_parameter<int>("stride", 1);
    const double rate_hz = declare_parameter<double>("rate_hz", 200.0);
    const std::string odom_topic =
        declare_parameter<std::string>("base_odom_topic", "/odom");
    const std::string joint_topic =
        declare_parameter<std::string>("joint_state_topic", "/joint_states");

    if (config_path.empty()) {
      throw std::runtime_error("Parameter config_path is required.");
    }
    if (!(rate_hz > 0.0)) {
      throw std::runtime_error("rate_hz must be positive.");
    }
    if (stride_param <= 0) {
      throw std::runtime_error("stride must be positive.");
    }
    stride_ = static_cast<std::size_t>(stride_param);

    cfg_ = crocoddyl::contact_id_xml::load_config(config_path);
    contact_id_experiment::validate_required(cfg_);
    model_ = contact_id_experiment::load_floating_base_model(cfg_.robot);
    joints_ = contact_id_experiment::resolve_identified_joints(model_, cfg_);
    joint_names_ = actuated_joint_names(model_);
    logs_ = contact_id_experiment::load_contact_id_logs(model_, joints_, cfg_);

    row_ = start_idx_param >= 0 ? static_cast<std::size_t>(start_idx_param)
                                : cfg_.solver.start_idx;
    if (row_ >= logs_.rows()) {
      throw std::runtime_error("Replay start_idx is outside the loaded logs.");
    }
    end_row_ = logs_.rows();
    if (max_samples_param > 0) {
      end_row_ = std::min(end_row_, row_ + static_cast<std::size_t>(max_samples_param));
    }

    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(odom_topic, 10);
    joint_pub_ = create_publisher<sensor_msgs::msg::JointState>(joint_topic, 10);

    timer_ = create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / rate_hz)),
        std::bind(&PrimeCsvReplayNode::timerCallback, this));

    RCLCPP_INFO(get_logger(), "Replaying rows [%zu, %zu) at %.2f Hz stride=%zu.",
                row_, end_row_, rate_hz, stride_);
  }

 private:
  void timerCallback() {
    if (row_ >= end_row_) {
      RCLCPP_INFO(get_logger(), "CSV replay finished.");
      timer_->cancel();
      return;
    }

    const rclcpp::Time stamp = now();
    const Eigen::VectorXd q = contact_id_experiment::csv_row(
        logs_.q_processed, row_, cfg_.data.q_has_time_column, model_.nq);
    const Eigen::VectorXd v = contact_id_experiment::csv_row(
        logs_.v_processed, row_, cfg_.data.v_has_time_column, model_.nv);
    const Eigen::Index u_offset = cfg_.data.u_has_time_column ? 1 : 0;
    const Eigen::VectorXd tau = logs_.u_processed
                                    .row(static_cast<Eigen::Index>(row_))
                                    .segment(u_offset, model_.nv - 6)
                                    .transpose();

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "world";
    odom.child_frame_id = "base";
    odom.pose.pose.position.x = q[0];
    odom.pose.pose.position.y = q[1];
    odom.pose.pose.position.z = q[2];
    odom.pose.pose.orientation.x = q[3];
    odom.pose.pose.orientation.y = q[4];
    odom.pose.pose.orientation.z = q[5];
    odom.pose.pose.orientation.w = q[6];
    odom.twist.twist.linear.x = v[0];
    odom.twist.twist.linear.y = v[1];
    odom.twist.twist.linear.z = v[2];
    odom.twist.twist.angular.x = v[3];
    odom.twist.twist.angular.y = v[4];
    odom.twist.twist.angular.z = v[5];

    sensor_msgs::msg::JointState joints;
    joints.header.stamp = stamp;
    joints.name = joint_names_;
    joints.position.resize(joint_names_.size());
    joints.velocity.resize(joint_names_.size());
    joints.effort.resize(joint_names_.size());
    for (std::size_t i = 0; i < joint_names_.size(); ++i) {
      joints.position[i] = q[7 + static_cast<Eigen::Index>(i)];
      joints.velocity[i] = v[6 + static_cast<Eigen::Index>(i)];
      joints.effort[i] = tau[static_cast<Eigen::Index>(i)];
    }

    odom_pub_->publish(odom);
    joint_pub_->publish(joints);
    row_ += stride_;
  }

  crocoddyl::ContactIDXMLConfig cfg_;
  pinocchio::Model model_;
  std::vector<pinocchio::JointIndex> joints_;
  std::vector<std::string> joint_names_;
  contact_id_experiment::LoadedContactIDLogs logs_;
  std::size_t row_;
  std::size_t end_row_;
  std::size_t stride_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<PrimeCsvReplayNode>());
  } catch (const std::exception& e) {
    std::cerr << "prime_csv_replay_node: " << e.what() << std::endl;
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
