import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)



panda = example_robot_data.load("panda")
robot_model = panda.model
state = crocoddyl.StateMultibody(robot_model)
actuation = crocoddyl.ActuationModelFull(state)
q0 = panda.model.referenceConfigurations["default"]
x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])

print("nq = ", robot_model.nq)
print("nv = ", robot_model.nv)

for i, name in enumerate(robot_model.names):
    print(f"{i}: {name}")
    
print("---------------")
for i in range(len(robot_model.joints)):
    joint = robot_model.joints[i]
    name = robot_model.names[i]
    print(f"{i}: {name} ({joint.nq} DoFs)")

print("x0 = ", x0)

nu = state.nv
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

x_des = x0.copy()
x_des[1] += 2.2
x_des[3] += 0.5
x_des[5] += 1.6

robot_data = robot_model.createData()
pinocchio.forwardKinematics(robot_model, robot_data, x_des[:robot_model.nq])
pinocchio.updateFramePlacements(robot_model, robot_data)
hand_id = robot_model.getFrameId("panda_hand")
frame_SE3 = robot_data.oMf[hand_id]

# Print translation and rotation
print("Translation:", frame_SE3.translation)
print("Rotation:\n", frame_SE3.rotation)

framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
    state,
    robot_model.getFrameId("panda_hand"),
    pinocchio.SE3(np.array([[1.0, 0.0, 0.0],[0.0, -1.0, 0.0],[0.0, 0.0, -1.0]]), np.array([0.8, 0.0, -0.2])),
    nu,
)

stateWeights = np.array( [1] * robot_model.nq + [1] * robot_model.nv) 
stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights)
            
uResidual = crocoddyl.ResidualModelControl(state, nu)
xResidual = crocoddyl.ResidualModelState(state, x_des, nu)
gravityResidual = crocoddyl.ResidualModelControlGrav(state, nu)

goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)

# xRegCost = crocoddyl.CostModelResidual(state, stateActivation, xResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
gravityRegCost = crocoddyl.CostModelResidual(state, gravityResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e1)
runningCostModel.addCost("xReg", xRegCost, 1e1)
# runningCostModel.addCost("uReg", uRegCost, 1e-1)
runningCostModel.addCost("gReg", gravityRegCost, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
# terminalCostModel.addCost("xReg", xRegCost, 1e3)

# Next, we need to create an action model for running and terminal knots. The
# forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFreeFwdDynamics.
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, runningCostModel
    ),
    dt,
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, terminalCostModel
    ),
    0.0,
)

# For this optimal control problem, we define 100 knots (or running action
# models) plus a terminal knot
T = 250
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverFDDP(problem)
if WITHPLOT:
    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving it with the solver algorithm
solver.solve()

print(
    "Finally reached = ",
    solver.problem.terminalData.differential.multibody.pinocchio.oMf[
        robot_model.getFrameId("panda_hand")
    ].translation.T,
)

# Plotting the solution and the solver convergence
if WITHPLOT:
    log = solver.getCallbacks()[1]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(
        log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2
    )

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    try:
        import gepetto

        cameraTF = [2.0, 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
        gepetto.corbaserver.Client()
        display = crocoddyl.GepettoDisplay(panda, 4, 4, cameraTF, floor=False)
    except Exception:
        display = crocoddyl.MeshcatDisplay(panda)

    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)