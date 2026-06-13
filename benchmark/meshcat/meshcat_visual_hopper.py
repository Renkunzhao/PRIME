import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import meshcat.geometry as g

# ---------------------------
# 1) Load URDF with FREE-FLYER
# ---------------------------
URDF_PATH = "/home/jkang/third_party/PRIME/data_hopper/hopper.urdf"
MESH_DIRS = []

# Force a floating base regardless of URDF contents
ff = pin.JointModelFreeFlyer()
model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, MESH_DIRS, ff)
data = model.createData()
nq, nv = model.nq, model.nv

# ---------------------------
# 2) Start Meshcat and load robot
# ---------------------------
viewer = meshcat.Visualizer()
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(viewer, open=True)
viz.loadViewerModel("hopper")
viz.display(pin.neutral(model))

# Prefer a frame named "base_link" if present, else use free-flyer joint (id=1)
try:
    base_frame_id = model.getFrameId("base_link")
    use_frame = True
except Exception:
    base_frame_id = 1
    use_frame = False

# ---------------------------
# 3) Load trajectory (rows = configurations)
#    Expect length == model.nq: [x,y,z,qx,qy,qz,qw, q_j1, q_j2, ...]
# ---------------------------
csv_A = "/home/jkang/third_party/PRIME/data_hopper/xs_rollout.csv"
# csv_A = "/home/jkang/third_party/PRIME/data_hopper/xs_log.csv"

Qraw = np.atleast_2d(np.loadtxt(csv_A, delimiter=","))

if Qraw.shape[1] < nq:
    raise ValueError(
        f"CSV has {Qraw.shape[1]} columns but model.nq = {nq}. "
        "For free-flyer, CSV must include base (xyz + quaternion) + joints."
    )

q_seq = Qraw[:, :nq]
T = len(q_seq)

# Diagnostics: show shapes and first two base poses
print(f"[info] model.nq = {nq}, CSV cols = {Qraw.shape[1]}, T = {T}")
print("[info] first base (xyz, quat):",
      q_seq[0, :3], q_seq[0, 3:7] if nq >= 7 else "(n/a)")
if T > 1:
    print("[info] second base (xyz, quat):",
          q_seq[1, :3], q_seq[1, 3:7] if nq >= 7 else "(n/a)")

# ---------------------------
# 4) Prepare base axes and trajectory once
# ---------------------------
axes_path = "base_axes"
traj_path = "base_traj"

# Small axes triad
L = 0.1
axes_pts = np.array([
    [0, 0, 0], [L, 0, 0],   # X
    [0, 0, 0], [0, L, 0],   # Y
    [0, 0, 0], [0, 0, L],   # Z
]).T
viewer[axes_path].set_object(g.Line(g.PointsGeometry(axes_pts)))
viewer[axes_path].set_transform(np.eye(4))

# Placeholder trajectory (replaced as we go)
viewer[traj_path].set_object(g.Line(g.PointsGeometry(np.zeros((3, 2)))))

traj_points = []

def set_frame(path, H):
    viewer[path].set_transform(H)

def draw_traj(path, pts_xyz):
    if len(pts_xyz) < 2:
        return
    P = np.array(pts_xyz).T  # (3, N)
    viewer[path].set_object(g.Line(g.PointsGeometry(P)))

# ---------------------------
# 5) Animate
# ---------------------------
rate_hz = 200.0
dt = 1.0 / rate_hz
idx = 0

for idx in range(T):
# for idx in range(10, 20):
    q = np.asarray(q_seq[idx], dtype=np.float64)

    # Display robot
    viz.display(q)

    # Base pose
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    if use_frame:
        H_base = data.oMf[base_frame_id].homogeneous
    else:
        H_base = data.oMi[1].homogeneous  # free-flyer joint pose in world

    # Move axes to base
    set_frame(axes_path, H_base)

    # Update trajectory (downsample drawing to save bandwidth)
    traj_points.append(H_base[:3, 3].copy())
    if idx % 5 == 0:
        draw_traj(traj_path, traj_points)

    idx = (idx + 1) % T
    time.sleep(1 * dt)