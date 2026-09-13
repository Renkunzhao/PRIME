#!/usr/bin/python3
"""Resample Go2 state at 200 Hz for PRIME's interval=0.005 configuration."""

import csv
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from unitree_go.msg import LowState

JOINT_NAMES = [
    f"{leg}_{joint}_joint"
    for leg in ("FR", "FL", "RR", "RL")
    for joint in ("hip", "thigh", "calf")
]


def make_state(low, estimate, stamp):
    """Use IEKF pose/world velocity and LowState joints; emit body-frame twist."""
    motors = low.motor_state[:12]
    orientation = estimate.pose.pose.orientation
    quat = np.array([orientation.w, orientation.x, orientation.y, orientation.z])
    if len(motors) != 12 or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
        raise ValueError("Invalid Go2 joint count or IEKF quaternion")
    w, x, y, z = quat / np.linalg.norm(quat)
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    position = estimate.pose.pose.position
    velocity = estimate.twist.twist.linear
    angular = estimate.twist.twist.angular
    body_velocity = rotation.T @ np.array([velocity.x, velocity.y, velocity.z])
    values = [
        position.x,
        position.y,
        position.z,
        *body_velocity,
        angular.x,
        angular.y,
        angular.z,
    ]
    values += [
        value for motor in motors for value in (motor.q, motor.dq, motor.tau_est)
    ]
    if not np.isfinite(values).all():
        raise ValueError("Go2 state contains NaN or infinity")
    odom = Odometry()
    odom.header.stamp = stamp
    odom.header.frame_id = "odom"
    odom.child_frame_id = "base"
    odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = (
        position.x,
        position.y,
        position.z,
    )
    odom.pose.pose.orientation.w = float(w)
    odom.pose.pose.orientation.x = float(x)
    odom.pose.pose.orientation.y = float(y)
    odom.pose.pose.orientation.z = float(z)
    odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = (
        map(float, body_velocity)
    )
    (
        odom.twist.twist.angular.x,
        odom.twist.twist.angular.y,
        odom.twist.twist.angular.z,
    ) = (angular.x, angular.y, angular.z)
    joints = JointState()
    joints.header.stamp = stamp
    joints.name = JOINT_NAMES
    joints.position = [float(motor.q) for motor in motors]
    joints.velocity = [float(motor.dq) for motor in motors]
    joints.effort = [float(motor.tau_est) for motor in motors]
    return odom, joints


class UnitreePrimeState(Node):
    def __init__(self):
        super().__init__("unitree_prime_state")
        self.csv_files = []
        self.q_writer = self.v_writer = self.tau_writer = None
        self.configure_csv_logging()
        self.low = self.estimate = None
        self.low_time = self.estimate_time = 0.0
        self.last_low_tick = None
        self.odom_publisher = self.create_publisher(Odometry, "/prime/odom", 10)
        self.joint_publisher = self.create_publisher(
            JointState, "/prime/joint_states", 10
        )
        self.low_subscription = self.create_subscription(
            LowState, "/lowstate", self.on_low, qos_profile_sensor_data
        )
        self.estimate_subscription = self.create_subscription(
            Odometry, "/iekf/odom", self.on_estimate, qos_profile_sensor_data
        )
        self.timer = self.create_timer(0.005, self.publish_state)

    def configure_csv_logging(self):
        self.declare_parameter("log_csv", False)
        self.declare_parameter("log_directory", "")
        self.declare_parameter("truncate_logs", True)
        if not self.get_parameter("log_csv").value:
            return

        directory_value = self.get_parameter("log_directory").value
        if not directory_value:
            raise ValueError("log_directory must be set when log_csv is enabled")
        directory = Path(directory_value).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        mode = "w" if self.get_parameter("truncate_logs").value else "a"
        paths = [
            directory / "p_sense.csv",
            directory / "v_sense.csv",
            directory / "tau_sense.csv",
        ]
        self.csv_files = [path.open(mode, newline="", buffering=1) for path in paths]
        self.q_writer, self.v_writer, self.tau_writer = [
            csv.writer(file) for file in self.csv_files
        ]
        self.get_logger().info(f"Writing PRIME-compatible CSV logs to {directory}")

    def on_low(self, message):
        if message.tick != self.last_low_tick:
            self.low, self.low_time = message, time.monotonic()
            self.last_low_tick = message.tick

    def on_estimate(self, message):
        # Track reception freshness for the 200 Hz latest-value resampler.
        self.estimate, self.estimate_time = message, time.monotonic()

    def publish_state(self):
        now = time.monotonic()
        if self.low is None or self.estimate is None:
            return
        if now - min(self.low_time, self.estimate_time) > 0.1:
            self.get_logger().warning(
                "LowState or IEKF odometry stale (>0.1 s); paused PRIME input",
                throttle_duration_sec=2.0,
            )
            return
        try:
            odom, joints = make_state(
                self.low, self.estimate, self.get_clock().now().to_msg()
            )
        except ValueError as error:
            self.get_logger().error(str(error), throttle_duration_sec=2.0)
            return
        self.odom_publisher.publish(odom)
        self.joint_publisher.publish(joints)
        self.write_csv_rows(odom, joints)

    def write_csv_rows(self, odom, joints):
        if self.q_writer is None:
            return
        stamp = odom.header.stamp
        stamp_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        linear = odom.twist.twist.linear
        angular = odom.twist.twist.angular
        self.q_writer.writerow(
            [
                stamp_ns,
                position.x,
                position.y,
                position.z,
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
                *joints.position,
            ]
        )
        self.v_writer.writerow(
            [
                stamp_ns,
                linear.x,
                linear.y,
                linear.z,
                angular.x,
                angular.y,
                angular.z,
                *joints.velocity,
            ]
        )
        self.tau_writer.writerow([stamp_ns, *joints.effort])

    def destroy_node(self):
        for file in self.csv_files:
            file.close()
        self.csv_files = []
        return super().destroy_node()


def main():
    rclpy.init()
    node = UnitreePrimeState()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
