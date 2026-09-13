#!/usr/bin/python3
"""Convert PRIME root inertia estimates to the Beam actor's raw 10D p."""

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Float32MultiArray, Float64MultiArray


def log_cholesky(dynamic):
    """Pinocchio [m,hx,hy,hz,Ixx,Ixy,Iyy,Ixz,Iyz,Izz] -> Beam theta.

    Input inertia is about the base origin; h = mass * COM. All quantities
    must use the training base frame and SI units.
    """
    dynamic = np.asarray(dynamic, dtype=np.float64)
    if dynamic.shape != (10,) or not np.isfinite(dynamic).all():
        raise ValueError("Expected exactly 10 finite dynamic inertia parameters")
    mass = dynamic[0]
    if mass <= 0:
        raise ValueError("Inertial mass must be positive")
    com = dynamic[1:4] / mass
    inertia_origin = dynamic[[4, 5, 7, 5, 6, 8, 7, 8, 9]].reshape(3, 3)
    inertia_com = inertia_origin - mass * (
        np.dot(com, com) * np.eye(3) - np.outer(com, com)
    )
    sigma = (0.5 * np.trace(inertia_com) * np.eye(3) - inertia_com) / mass
    # Reverse-axis Cholesky gives U with sigma = U @ U.T, not U.T @ U.
    try:
        upper = np.linalg.cholesky(sigma[::-1, ::-1])[::-1, ::-1]
    except np.linalg.LinAlgError as error:
        raise ValueError("Inertia has non-positive COM second moment") from error
    theta = np.array(
        [
            0.5 * np.log(mass),
            *np.log(np.diag(upper)),
            upper[0, 1],
            upper[1, 2],
            upper[0, 2],
            *com,
        ]
    )
    if not np.isfinite(theta).all():
        raise ValueError("Inertial conversion produced non-finite parameters")
    return theta


class PrimeInertialInput(Node):
    def __init__(self):
        super().__init__("prime_inertial_input")
        self.declare_parameter(
            "policy_nominal_dynamic_parameters", Parameter.Type.DOUBLE_ARRAY
        )
        self.policy_nominal = np.asarray(
            self.get_parameter("policy_nominal_dynamic_parameters").value,
            dtype=np.float64,
        )
        self.policy_theta = log_cholesky(self.policy_nominal)
        self.prime_nominal = None
        self.publisher = self.create_publisher(
            Float32MultiArray, "/prime/inertial_delta", 1
        )
        self.nominal_subscription = self.create_subscription(
            Float64MultiArray,
            "/prime_moving_window_node/nominal_inertia_dynamic_params",
            self.on_nominal,
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.estimate_subscription = self.create_subscription(
            Float64MultiArray,
            "/prime_moving_window_node/inertia_dynamic_params",
            self.on_estimate,
            1,
        )

    def on_nominal(self, message):
        nominal = np.asarray(message.data, dtype=np.float64)
        log_cholesky(nominal)  # Fail startup/reconfiguration for an invalid model.
        self.prime_nominal = nominal
        self.get_logger().info(
            f"PRIME nominal mass {nominal[0]:.6f} kg; "
            f"policy nominal mass {self.policy_nominal[0]:.6f} kg"
        )

    def on_estimate(self, message):
        if self.prime_nominal is None:
            self.get_logger().warning(
                "Waiting for PRIME model nominal", throttle_duration_sec=5.0
            )
            return
        try:
            estimate = np.asarray(message.data, dtype=np.float64)
            log_cholesky(estimate)
            # Remove the fixed model difference in additive physical parameters,
            # before applying the nonlinear Log-Cholesky transform.
            policy_inertia = self.policy_nominal + (estimate - self.prime_nominal)
            delta = log_cholesky(policy_inertia) - self.policy_theta
            if np.any(np.abs(delta) > np.finfo(np.float32).max):
                raise ValueError("Inertial delta cannot be represented as float32")
        except ValueError as error:
            self.get_logger().error(
                f"Rejected PRIME estimate: {error}", throttle_duration_sec=2.0
            )
            return
        self.publisher.publish(
            Float32MultiArray(data=delta.astype(np.float32).tolist())
        )


def main():
    rclpy.init()
    node = None
    try:
        node = PrimeInertialInput()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
