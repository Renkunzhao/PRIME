import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import pandas as pd

import crocoddyl
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the anymal model
# anymal = example_robot_data.load("anymal")
# anymal = example_robot_data.load("hyq")
anymal = example_robot_data.load("go1")

# Defining the initial state of the robot
q0 = anymal.model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(anymal.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
# lfFoot, rfFoot, lhFoot, rhFoot = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
# lfFoot, rfFoot, lhFoot, rhFoot = "lf_foot", "rf_foot","lh_foot", "rh_foot"
rfFoot, lfFoot, rhFoot , lhFoot = "FR_foot_fixed", "FL_foot_fixed","RR_foot_fixed", "RL_foot_fixed"

gait = SimpleQuadrupedalGaitProblem(anymal.model, rfFoot, lfFoot, rhFoot , lhFoot)

# Setting up all tasks
GAITPHASES = [
    # {
    #     "walking": {
    #         "stepLength": 0.25,
    #         "stepHeight": 0.15,
    #         "timeStep": 1e-2,
    #         "stepKnots": 25,
    #         "supportKnots": 2,
    #     }
    # },
    {
        "trotting": {
            "stepLength": 0.15,
            "stepHeight": 0.1,
            "timeStep": 2e-3,
            "stepKnots": 50,
            "supportKnots": 10,
        }
    },
    {
        "trotting": {
            "stepLength": 0.15,
            "stepHeight": 0.1,
            "timeStep": 2e-3,
            "stepKnots": 50,
            "supportKnots": 10,
        }
    },
    # {
    #     "trotting": {
    #         "stepLength": 0.15,
    #         "stepHeight": 0.1,
    #         "timeStep": 1e-2,
    #         "stepKnots": 25,
    #         "supportKnots": 2,
    #     }
    # },
    # {
    #     "trotting": {
    #         "stepLength": 0.15,
    #         "stepHeight": 0.1,
    #         "timeStep": 1e-2,
    #         "stepKnots": 25,
    #         "supportKnots": 2,
    #     }
    # },
    # {
    #     "trotting": {
    #         "stepLength": 0.15,
    #         "stepHeight": 0.1,
    #         "timeStep": 1e-2,
    #         "stepKnots": 25,
    #         "supportKnots": 2,
    #     }
    # },
    # {
    #     "pacing": {
    #         "stepLength": 0.15,
    #         "stepHeight": 0.1,
    #         "timeStep": 1e-2,
    #         "stepKnots": 25,
    #         "supportKnots": 5,
    #     }
    # },
    # {
    #     "bounding": {
    #         "stepLength": 0.15,
    #         "stepHeight": 0.1,
    #         "timeStep": 1e-2,z
    #         "stepKnots": 25,
    #         "supportKnots": 5,
    #     }
    # },
    # {
    #     "jumping": {
    #         "jumpHeight": 0.15,
    #         "jumpLength": [0.0, 0.3, 0.0],
    #         "timeStep": 1e-2,
    #         "groundKnots": 10,
    #         "flyingKnots": 20,
    #     }
    # },
]

xs_list = []
us_list = []

expected_x_dim = None
expected_u_dim = None

solver = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "walking":
            # Creating a walking problem
            solver[i] = crocoddyl.SolverFDDP(
                gait.createWalkingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
        elif key == "trotting":
            # Creating a trotting problem
            solver[i] = crocoddyl.SolverFDDP(
                gait.createTrottingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
        elif key == "pacing":
            # Creating a pacing problem
            solver[i] = crocoddyl.SolverFDDP(
                gait.createPacingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
        elif key == "bounding":
            # Creating a bounding problem
            solver[i] = crocoddyl.SolverFDDP(
                gait.createBoundingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
        elif key == "jumping":
            # Creating a jumping problem
            solver[i] = crocoddyl.SolverFDDP(
                gait.createJumpingProblem(
                    x0,
                    value["jumpHeight"],
                    value["jumpLength"],
                    value["timeStep"],
                    value["groundKnots"],
                    value["flyingKnots"],
                )
            )

    # Added the callback functions
    print("*** SOLVE " + key + " ***")
    if WITHPLOT:
        solver[i].setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
    else:
        solver[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the DDP solver
    xs = [x0] * (solver[i].problem.T + 1)
    us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
    solver[i].solve(xs, us, 100, False)

    # Defining the final state as initial one for the next phase
    x0 = solver[i].xs[-1]

    for x in solver[i].xs:
        x_arr = np.asarray(x).flatten()
        if expected_x_dim is None:
            expected_x_dim = x_arr.shape[0]
        if x_arr.size == 0:
            x_arr = np.zeros(expected_x_dim)
        xs_list.append(x_arr)

    for u in solver[i].us:
        u_arr = np.asarray(u).flatten()
        if expected_u_dim is None:
            expected_u_dim = u_arr.shape[0] if u_arr.size > 0 else 1  # Default to 1 if all empty
        if u_arr.size == 0:
            u_arr = np.zeros(expected_u_dim)
        us_list.append(u_arr)

xs_all_np = np.vstack(xs_list)
us_all_np = np.vstack(us_list)

import os
print(os.getcwd())  # 确认当前目录

np.savetxt("xs_vectors.csv", xs_all_np, delimiter=",")
# np.savetxt("xs_vectors.csv", xs_all_np, delimiter=",")
np.savetxt("us_vectors.csv", us_all_np, delimiter=",")
print("Data saved to 'solver_results.xlsx'")

# Display the entire motion
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        cameraTF = [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
        display = crocoddyl.GepettoDisplay(anymal, 4, 4, cameraTF)
    except Exception:
        display = crocoddyl.MeshcatDisplay(anymal)
    display.rate = -1
    display.freq = 1
    while True:
        for i, phase in enumerate(GAITPHASES):
            display.displayFromSolver(solver[i])
        time.sleep(1.0)

# Plotting the entire motion
if WITHPLOT:
    plotSolution(solver, figIndex=1, show=False)

    for i, phase in enumerate(GAITPHASES):
        title = next(iter(phase.keys())) + " (phase " + str(i) + ")"
        log = solver[i].getCallbacks()[1]
        crocoddyl.plotConvergence(
            log.costs,
            log.pregs,
            log.dregs,
            log.grads,
            log.stops,
            log.steps,
            figTitle=title,
            figIndex=i + 3,
            show=True if i == len(GAITPHASES) - 1 else False,
        )
