# import numpy as np
# import pinocchio
# import example_robot_data
# from pinocchio.visualize import MeshcatVisualizer
# import meshcat
# import time
# import os

# # Load model
# robot = example_robot_data.load("panda")
# model = robot.model
# # viz = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
# # viz.initViewer(open=True)
# # viz.loadViewerModel()
# meshcat_viz = meshcat.Visualizer()

# viz1 = MeshcatVisualizer(model, robot.collision_model, robot.visual_model)
# viz1.initViewer(meshcat_viz, open=False)  # Here you pass your own viewer!
# viz1.loadViewerModel("arm_fddp")


# # Load trajectory from CSV
# q_data = np.loadtxt("/home/jkang/third_party/PRIME/data_RRR/xs_results_fddp.csv", delimiter=",")
# start_idx = 0
# n_knots = 600
# n_scale = 1
# end_idx = start_idx + n_knots

# # Play animation
# while True:
#     for q_fddp in zip(q_data):
#         q_fddp = np.array(q_fddp, dtype=np.float64).reshape(-1)[:9]
#         viz1.display(q_fddp)
#         time.sleep(0.01)
#     time.sleep(2)



import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import meshcat.transformations as tf

# ---------------------------
# 1) Load your URDF (primitives OK)
# ---------------------------
URDF_PATH = "/home/jkang/third_party/PRIME/data_RRR/RRR_arm.urdf"
MESH_DIRS = []  # no meshes needed for primitive-only URDF

# Build model + geom models once
model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, MESH_DIRS)

# ---------------------------
# 2) Start Meshcat and load TWO robot instances
# ---------------------------
viewer = meshcat.Visualizer()

vizA = MeshcatVisualizer(model, collision_model, visual_model)
vizB = MeshcatVisualizer(model, collision_model, visual_model)

vizA.initViewer(viewer, open=True)
vizB.initViewer(viewer, open=False)

vizA.loadViewerModel("armA")
vizB.loadViewerModel("armB")

# Optional: move armB aside so they don't overlap (translate +X by 0.6 m)
T_armB = tf.translation_matrix([0.6, 0.0, 0.0])
viewer["armB"].set_transform(T_armB)

# Show neutral pose initially
q0 = pin.neutral(model)
vizA.display(q0)
vizB.display(q0)

# ---------------------------
# 3) Load two trajectories
#    (each row = a configuration; must have >= model.nq columns)
# ---------------------------
csv_A = "/home/jkang/third_party/PRIME/data_RRR/xs_rollout.csv"
csv_B = "/home/jkang/third_party/PRIME/data_RRR/xs_sim.csv"

q_data_A = np.loadtxt(csv_A, delimiter=",")
q_data_B = np.loadtxt(csv_B, delimiter=",")

nq = model.nq
q_data_A = np.atleast_2d(q_data_A)[:, :nq]
q_data_B = np.atleast_2d(q_data_B)[:, :nq]

TA, TB = len(q_data_A), len(q_data_B)

# ---------------------------
# 4) Play both animations together
# ---------------------------
rate_hz = 100.0
dt = 1.0 / rate_hz

idxA, idxB = 0, 0
loop = True

while loop:
    # display current frames
    vizA.display(q_data_A[idxA])
    vizB.display(q_data_B[idxB])

    # advance (wrap independently so lengths can differ)
    idxA = (idxA + 1) % TA
    idxB = (idxB + 1) % TB

    time.sleep(dt)