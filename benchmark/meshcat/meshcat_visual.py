import numpy as np
import pinocchio
import example_robot_data
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import time
import os

def display_only_transforms(viz, q):
    data = pinocchio.Data(model)
    pinocchio.forwardKinematics(model, data, q)
    for o in robot.visual_model.geometryObjects:
        name = f"go1_sim/{o.name}"
        M = data.oMi[o.parentJoint] * o.placement
        viz.viewer[name].set_transform(M.homogeneous)  

# Load model
robot = example_robot_data.load("go1")
model = robot.model
# viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
# viz.initViewer(open=True)
# viz.loadViewerModel()
meshcat_viz = meshcat.Visualizer()

viz1 = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
viz1.initViewer(meshcat_viz, open=False)  # Here you pass your own viewer!
viz1.loadViewerModel("go1_fddp")

viz2 = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
viz2.initViewer(meshcat_viz, open=False)
viz2.loadViewerModel("go1_sim")

# Load trajectory from CSV
q_data = np.loadtxt("/home/jkang/third_party/PRIME/data_go1_rl/xs_rollout.csv", delimiter=",")
q_data_sim = np.loadtxt("/home/jkang/third_party/PRIME/data_go1_rl/xs_sim.csv", delimiter=",")
start_idx = 100
n_knots = 1000
n_scale =10
end_idx = start_idx + n_knots
q_sampled = q_data_sim[start_idx:end_idx:n_scale]
# q_sampled = q_data_sim
# Play animation
while True:
    for q_fddp, q_sim in zip(q_data, q_sampled):
        viz1.display(q_fddp[:19])
        viz2.display(q_sim[:19])
        time.sleep(0.02)
    time.sleep(2)
# while True:
#     for q_sim in q_data_sim:
#         viz2.display(q_sim[:19])
#         time.sleep(0.001)
#     time.sleep(2)
