import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import meshcat.geometry as g
from meshcat.geometry import MeshPhongMaterial, StlMeshGeometry, DaeMeshGeometry

# ---------------------------
# MeshCat Viewer + simple floor
# ---------------------------
viewer = meshcat.Visualizer()
viewer["/Background"].set_property("top_color", [1.0, 1.0, 1.0])
viewer["/Background"].set_property("bottom_color", [0.7, 0.7, 0.7])
viewer["/Grid"].set_property("visible", True)

floor = g.Box([20.0, 20.0, 0.02])
viewer["floor"].set_object(floor, MeshPhongMaterial(color=0x999999, specular=0x111111, shininess=30))
T = np.eye(4); T[2, 3] = -0.01
viewer["floor"].set_transform(T)

# ---------------------------
# Paths
# ---------------------------
URDF_PATH = "/home/jkang/third_party/PRIME/data_b1_real/urdf/B1.urdf"
MESH_DIRS = "/home/jkang/third_party/PRIME/data_b1_real/"

CSV_FILES = [
    "/home/jkang/third_party/PRIME/data_b1_real/log_csv/x_mocap.csv",  # traj A
    "/home/jkang/third_party/PRIME/data_b1_real/xs_results_fddp.csv",  # traj B
]

# ---------------------------
# Build one robot model (free-flyer)
# ---------------------------
ff = pin.JointModelFreeFlyer()
model, collision_model, visual_model = pin.buildModelsFromUrdf(URDF_PATH, MESH_DIRS, ff)

# ---------------------------
# Create 2 visualizers (same model), different root names/colors
# ---------------------------
def make_viz(root_name: str, color: int) -> MeshcatVisualizer:
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(viewer, open=False)
    viz.loadViewerModel(root_name)
    # Recolor visuals for this instance
    for geom in viz.visual_model.geometryObjects:
        path = viz.getViewerNodeName(geom, "visual")
        if path is None:
            continue
        if geom.meshPath.endswith(".dae"):
            obj = DaeMeshGeometry.from_file(geom.meshPath)
        elif geom.meshPath.endswith(".stl"):
            obj = StlMeshGeometry.from_file(geom.meshPath)
        else:
            continue
        viz.viewer[path].set_object(obj, MeshPhongMaterial(color=color, transparent=True, opacity=0.6))
    return viz

viz_list = [
    make_viz("B1_A", 0xFF4040),  # red-ish
    make_viz("B1_B", 0x4040FF),  # blue-ish
]

# ---------------------------
# Load both trajectories (assumes columns: [px py pz qw qx qy qz 12*joints ...])
# ---------------------------
def load_qseq(csv_path: str, nq: int) -> np.ndarray:
    Qraw = np.atleast_2d(np.loadtxt(csv_path, delimiter=","))
    if Qraw.shape[1] < nq:
        raise ValueError(f"{csv_path}: CSV has {Qraw.shape[1]} columns but nq={nq}")
    # Your layout (adjust if needed):
    p_base = Qraw[:, 0:3]
    q_base = Qraw[:, 3:7]     # [w x y z]
    joints = Qraw[:, 7:19]    # 12 joints
    return np.hstack([p_base, q_base, joints])

down_sample = 4
start_idx = 9800
qseq_list = [load_qseq(csv, model.nq) for csv in CSV_FILES]
# qseq_list[0] = qseq_list[0][::down_sample]
qseq_list[0] = qseq_list[0][start_idx::down_sample]

# Optional: spatially separate them a bit so you can see both clearly
x_offsets = [0.0, 0.5]  # shift traj B by +0.5m in x
for qseq, xoff in zip(qseq_list, x_offsets):
    qseq[:, 0] += xoff  # add to base x

# ---------------------------
# Animate both (use shortest length)
# ---------------------------
datas = [model.createData(), model.createData()]
T = min(len(q) for q in qseq_list)
rate_hz = 1000.0
dt = 1.0 / rate_hz

# Camera follows the first robot (optional)
follow_first = True
try:
    base_frame_id = model.getFrameId("base")
    use_frame = True
except Exception:
    base_frame_id = 1
    use_frame = False

while True:
    for t in range(T):
        # Display both
        for i, (viz, qseq) in enumerate(zip(viz_list, qseq_list)):
            q = np.asarray(qseq[t], dtype=np.float64)
            viz.display(q)
            if follow_first and i == 0:
                q0 = q

        if follow_first:
            pin.forwardKinematics(model, datas[0], q0)
            pin.updateFramePlacements(model, datas[0])
            H_base = datas[0].oMf[base_frame_id].homogeneous if use_frame else datas[0].oMi[1].homogeneous
            offset = np.eye(4); offset[:3, 3] = [-0.8, 0.2, 0.6]
            H_cam = H_base @ offset
            # viewer["/Cameras/default"].set_transform(H_cam)  # uncomment to move camera

        time.sleep(down_sample * dt)