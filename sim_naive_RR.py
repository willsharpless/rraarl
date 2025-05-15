"""
Please contact the author(s) of this library if you have any questions.
Authors: Kai-Chieh Hsu ( kaichieh@princeton.edu )

This experiment runs double deep Q-network with the discounted reach-avoid
Bellman equation (DRABE) proposed in [RSS21] on a 2-dimensional point mass
problem. We use this script to generate Fig. 2 and Fig. 3 in the paper.

Examples:
    RA:
        python3 sim_naive.py -w -sf -of scratch -a -g 0.99 -n anneal
        python3 sim_naive.py -w -sf -of scratch -n 9999
        python3 sim_naive.py -w -sf -of scratch -g 0.999 -dt fail -n 999
    Lagrange:
        python3 sim_naive.py -sf -m lagrange -of scratch -g 0.95 -n 95
        python3 sim_naive.py -sf -m lagrange -of scratch -dt TF -g 0.95 -n 95
    test: python3 sim_naive.py -w -sf -of scratch -wi 100 -mu 100 -cp 40
"""

import os
import argparse
import time
from warnings import simplefilter
import gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

from RARL.DDQNSingleRRAA import DDQNSingleRRAA
from RARL.config import dqnConfig
from RARL.utils import save_obj
from gym_reachability import gym_reachability  # Custom Gym env.

matplotlib.use('Agg')
simplefilter(action='ignore', category=FutureWarning)
timestr = time.strftime("%Y-%m-%d-%H_%M")

# == ARGS ==
parser = argparse.ArgumentParser()

# environment parameters
parser.add_argument(
    "-dt", "--doneType", help="when to raise done flag", default='toEnd',
    type=str
)
parser.add_argument(
    "-ct", "--costType", help="cost type", default='sparse', type=str
)
parser.add_argument(
    "-rnd", "--randomSeed", help="random seed", default=0, type=int
)
parser.add_argument(
    "-r", "--reward", help="when entering target set", default=-1, type=float
)
parser.add_argument(
    "-p", "--penalty", help="when entering failure set", default=1, type=float
)
parser.add_argument(
    "-s", "--scaling", help="scaling of ell/g", default=4, type=float
)
parser.add_argument(
    "-et", "--envType", help="env type", default='R', type=str
)

# training scheme
parser.add_argument(
    "-w", "--warmup", help="warmup Q-network", action="store_true"
)
parser.add_argument(
    "-wi", "--warmupIter", help="warmup iteration", default=2000, type=int
)
parser.add_argument(
    "-mu", "--maxUpdates", help="maximal #gradient updates", default=400000,
    type=int
)
parser.add_argument(
    "-ut", "--updateTimes", help="#hyper-param. steps", default=10, type=int
)
parser.add_argument(
    "-mc", "--memoryCapacity", help="memoryCapacity", default=10000, type=int
)
parser.add_argument(
    "-cp", "--checkPeriod", help="check period", default=20000, type=int
)

# NN hyper-parameters
parser.add_argument(
    "-a", "--annealing", help="gamma annealing", action="store_true"
)
parser.add_argument(
    "-arc", "--architecture", help="NN architecture", default=[100, 20],
    nargs="*", type=int
)
parser.add_argument(
    "-lr", "--learningRate", help="learning rate", default=1e-3, type=float
)
parser.add_argument(
    "-g", "--gamma", help="contraction coeff.", default=0.9999, type=float
)
parser.add_argument(
    "-act", "--actType", help="activation type", default='Tanh', type=str
)

# RL type
parser.add_argument("-m", "--mode", help="mode", default='R', type=str)
parser.add_argument(
    "-tt", "--terminalType", help="terminal value", default='g', type=str
)

# file
parser.add_argument(
    "-st", "--showTime", help="show timestr", action="store_true"
)
parser.add_argument("-n", "--name", help="extra name", default='test', type=str)
parser.add_argument(
    "-of", "--outFolder", help="output file", default='scratch', type=str
)
parser.add_argument(
    "-pf", "--plotFigure", help="plot figures", action="store_true"
)
parser.add_argument(
    "-sf", "--storeFigure", help="store figures", action="store_true", default=True,
)

parser.add_argument(
    "-sl", "--skip_learn", help="skip learning and plot", action="store_true", default=False,
)

parser.add_argument(
    "-lp", "--load_path", help="path to trained model to restore", default='', type=str
)

parser.add_argument(
    "-ls", "--load_step", help="path to trained model to restore", default=40000, type=int
)

# load models
parser.add_argument(
    "-lp1", "--load_model_path_1", help="path to presovled Q network (1)", default='', type=str
)
parser.add_argument(
    "-lp2", "--load_model_path_2", help="path to presovled Q network (2)", default='', type=str
)

args = parser.parse_args()
print(args)

# # debugging
# args.mode = 'A'
# args.envType = 'gap_avoid'
# args.skip_learn = True
# args.load_path = './presolved/zermelo_show_RAA/A/avoid_mu1m-toEnd/'
# args.load_step = 400000

# args.mode = 'RAA'
# args.envType = 'gap'
# args.load_model_path_1 = './presolved/zermelo_show_RAA/A/avoid_mu1m-toEnd/Q-400000.pth'
# args.skip_learn = True
# args.load_path = './scratch/naiveRAA/DDQN/RAA/-toEnd/'
# args.load_step = 260000

# args.mode = 'RA'
# args.envType = 'gap'
# args.load_model_path_1 = './presolved/zermelo_show_RAA/A/avoid_mu1m-toEnd/Q-20000.pth'
# args.skip_learn = True
# args.load_path = './scratch/naiveRAA/DDQN/RA/naiveRAA_2-toEnd/'
# args.load_step = 280000

# args.mode = 'R'
# args.envType = 'R'

# args.name = 'R1_test'
# args.mode = 'R'
# args.envType = 'R1'

# args.name = 'R2_test'
# args.mode = 'R'
# args.envType = 'R2'

args.mode = 'RR'
args.envType = 'RR'
args.load_model_path_1 = './scratch/naiveRR/DDQN/R/R1_test-toEnd/model/Q-60000.pth'
args.load_model_path_2 = './scratch/naiveRR/DDQN/R/R2_test-toEnd/model/Q-400000.pth'

# == CONFIGURATION ==
env_name = "zermelo_show_RR"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxUpdates = args.maxUpdates
updateTimes = args.updateTimes
updatePeriod = int(maxUpdates / updateTimes)
maxSteps = 250
storeFigure = args.storeFigure
plotFigure = args.plotFigure

if args.mode == 'lagrange':
  fn = args.name + '-' + args.doneType + '-' + args.costType
else:
  fn = args.name + '-' + args.doneType
if args.showTime:
  fn = fn + '-' + timestr

outFolder = os.path.join(args.outFolder, 'naiveRR', 'DDQN', args.mode, fn)
print(outFolder)
figureFolder = os.path.join(outFolder, 'figure')
os.makedirs(figureFolder, exist_ok=True)

if args.mode == 'lagrange':
  envMode = 'normal'
  agentMode = 'normal'
  GAMMA_END = args.gamma
  EPS_PERIOD = updatePeriod
  EPS_RESET_PERIOD = maxUpdates
else:
  envMode = args.mode
  agentMode = args.mode
  if args.annealing:
    GAMMA_END = 0.999999
    EPS_PERIOD = int(updatePeriod / 10)
    EPS_RESET_PERIOD = updatePeriod
  else:
    GAMMA_END = args.gamma
    EPS_PERIOD = updatePeriod
    EPS_RESET_PERIOD = maxUpdates

if args.doneType == 'toEnd':
  sample_inside_obs = True
elif args.doneType == 'TF' or args.doneType == 'fail':
  sample_inside_obs = False

# == Environment ==
print("\n== Environment Information ==")
env = gym.make(
    env_name, device=device, mode=envMode, doneType=args.doneType,
    sample_inside_obs=sample_inside_obs, envType=args.envType
)

stateDim = env.state.shape[0]
actionNum = env.action_space.n
action_list = np.arange(actionNum)
print(
    "State Dimension: {:d}, ActionSpace Dimension: {:d}".format(
        stateDim, actionNum
    )
)
print(env.discrete_controls)

env.set_costParam(args.penalty, args.reward, args.costType, args.scaling)
env.set_seed(args.randomSeed)
print(
    "Cost type: {}, Margin scaling: {:.1f}, ".
    format(env.costType, env.scaling)
    + "Reward: {:.1f}, Penalty: {:.1f}".format(env.reward, env.penalty)
)

if plotFigure or storeFigure:
  nx, ny = 101, 101
  vmin = -1 * args.scaling
  vmax = 1 * args.scaling

  v = np.zeros((nx, ny))
  l_x = np.zeros((nx, ny))
  l2_x = np.zeros((nx, ny))
#   g_x = np.zeros((nx, ny))
  xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
  ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)

  it = np.nditer(v, flags=['multi_index'])
  
  while not it.finished:
    idx = it.multi_index
    x = xs[idx[0]]
    y = ys[idx[1]]

    l_x[idx] = env.target_margin(np.array([x, y]))
    l2_x[idx] = env.target2_margin(np.array([x, y]))

    v[idx] = np.maximum(l_x[idx], l2_x[idx])
    it.iternext()

  axStyle = env.get_axes()

  fig, axes = plt.subplots(1, 3, figsize=(12, 6))

  ax = axes[0]
  im = ax.imshow(
      l_x.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap="seismic", vmin=vmin, vmax=vmax
  )
  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
  )
  cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
  ax.set_title(r'$\ell(x)$', fontsize=18)

  ax = axes[1]
  im = ax.imshow(
      l2_x.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap="seismic", vmin=vmin, vmax=vmax
  )
  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
  )
  cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
  ax.set_title(r'$\ell_2(x)$', fontsize=18)

  ax = axes[2]
  im = ax.imshow(
      v.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap="seismic", vmin=vmin, vmax=vmax
  )
#   env.plot_reach_avoid_set(ax)
  cbar = fig.colorbar(
      im, ax=ax, pad=0.01, fraction=0.05, shrink=.95, ticks=[vmin, 0, vmax]
  )
  cbar.ax.set_yticklabels(labels=[vmin, 0, vmax], fontsize=24)
  ax.set_title(r'$v(x)$', fontsize=18)

  for ax in axes:
    env.plot_target_sets(ax=ax)
    env.plot_formatting(ax=ax)

  fig.tight_layout()
  if storeFigure:
    figurePath = os.path.join(figureFolder, 'env.png')
    fig.savefig(figurePath)
  if plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()

# == Agent CONFIG ==
print("\n== Agent Information ==")
CONFIG = dqnConfig(
    DEVICE=device, ENV_NAME=env_name, SEED=args.randomSeed,
    MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, BATCH_SIZE=64,
    MEMORY_CAPACITY=args.memoryCapacity, ARCHITECTURE=args.architecture,
    ACTIVATION=args.actType, GAMMA=args.gamma, GAMMA_PERIOD=updatePeriod,
    GAMMA_END=GAMMA_END, EPS_PERIOD=EPS_PERIOD, EPS_DECAY=0.7,
    EPS_RESET_PERIOD=EPS_RESET_PERIOD, LR_C=args.learningRate,
    LR_C_PERIOD=updatePeriod, LR_C_DECAY=0.8, MAX_MODEL=100,
    LOAD_MODEL_PATH_1=args.load_model_path_1,
    LOAD_MODEL_PATH_2=args.load_model_path_2,
)

# == AGENT ==
dimList = [stateDim] + CONFIG.ARCHITECTURE + [actionNum]
agent = DDQNSingleRRAA(
    CONFIG, actionNum, action_list, dimList=dimList, mode=agentMode,
    terminalType=args.terminalType
)
print("We want to use: {}, and Agent uses: {}".format(device, agent.device))
print("Critic is using cuda: ", next(agent.Q_network.parameters()).is_cuda)

if not args.skip_learn:
    if args.warmup:
        print("\n== Warmup Q ==")
        lossList = agent.initQ(
            env, args.warmupIter, outFolder, num_warmup_samples=200, vmin=vmin,
            vmax=vmax, plotFigure=plotFigure, storeFigure=storeFigure
        )

        if plotFigure or storeFigure:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            tmp = np.arange(500, args.warmupIter)
            # tmp = np.arange(args.warmupIter)
            ax.plot(tmp, lossList[tmp], 'b-')
            ax.set_xlabel('Iteration', fontsize=18)
            ax.set_ylabel('Loss', fontsize=18)
            plt.tight_layout()

        if storeFigure:
            figurePath = os.path.join(figureFolder, 'initQ_Loss.png')
            fig.savefig(figurePath)
        if plotFigure:
            plt.show()
            plt.pause(0.001)
            plt.close()

    print("\n== Training Information ==")
    vmin = -1 * args.scaling
    vmax = 1 * args.scaling
    checkPeriod = args.checkPeriod
    trainRecords, trainProgress = agent.learn(
        env, MAX_UPDATES=maxUpdates, MAX_EP_STEPS=maxSteps, warmupQ=False,
        doneTerminate=True, vmin=vmin, vmax=vmax, numRndTraj=10000,
        checkPeriod=checkPeriod, outFolder=outFolder, plotFigure=args.plotFigure,
        storeFigure=args.storeFigure
    )

    trainDict = {}
    trainDict['trainRecords'] = trainRecords
    trainDict['trainProgress'] = trainProgress
    filePath = os.path.join(outFolder, 'train')

else:
  agent.restore(args.load_step, args.load_path)

if plotFigure or storeFigure:
  # = loss
  if not args.skip_learn:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    data = trainRecords
    ax = axes[0]
    ax.plot(data, 'b:')
    ax.set_xlabel('Iteration (x 1e5)', fontsize=18)
    ax.set_xticks(np.linspace(0, maxUpdates, 5))
    ax.set_xticklabels(np.linspace(0, maxUpdates, 5) / 1e5)
    ax.set_title('loss_critic', fontsize=18)
    ax.set_xlim(left=0, right=maxUpdates)

    data = trainProgress[:, 0]
    ax = axes[1]
    x = np.arange(data.shape[0]) + 1
    ax.plot(x, data, 'b-o')
    ax.set_xlabel('Index', fontsize=18)
    ax.set_xticks(x)
    # ax.set_xticklabels(np.arange(data.shape[0]) + 1)
    ax.set_title('Success Rate', fontsize=18)
    ax.set_xlim(left=1, right=data.shape[0])

    fig.tight_layout()
    if storeFigure:
        figurePath = os.path.join(figureFolder, 'train_loss_success.png')
        fig.savefig(figurePath)
    if plotFigure:
        plt.show()
        plt.pause(0.001)
    plt.close()

  # = value_rollout_action
  if not args.skip_learn:
    idx = np.argmax(trainProgress[:, 0]) + 1
    successRate = np.amax(trainProgress[:, 0])
    print('We pick model with success rate-{:.3f}'.format(successRate))
    agent.restore(idx * args.checkPeriod, outFolder)
    restored_model_ix = idx * args.checkPeriod
  else:
    restored_model_ix = args.load_step

  nx = 41
  ny = 121
  xs = np.linspace(env.bounds[0, 0], env.bounds[0, 1], nx)
  ys = np.linspace(env.bounds[1, 0], env.bounds[1, 1], ny)
  
  if args.mode in ["RA", "R", "A"]:
    resultMtx = np.empty((nx, ny), dtype=int)
    actDistMtx = np.empty((nx, ny), dtype=int)
    it = np.nditer(resultMtx, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index
        print(idx, end='\r')
        x = xs[idx[0]]
        y = ys[idx[1]]

        state = np.array([x, y])
        stateTensor = torch.FloatTensor(state).to(agent.device).unsqueeze(0)
        action_index = agent.Q_network(stateTensor).min(dim=1)[1].cpu().item()
        # u = env.discrete_controls[action_index]
        actDistMtx[idx] = action_index

        _, _, result = env.simulate_one_trajectory(
            agent.Q_network, T=250, state=state, toEnd=False, q_func_1=agent.Q_decomposed_1, q_func_2=agent.Q_decomposed_2,
        )
        resultMtx[idx] = result
        it.iternext()

#   fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
  fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
  axStyle = env.get_axes()
  
  # = Rollout
  ax = axes[1]
#   im = ax.imshow(
#       resultMtx.T != 1, interpolation='none', extent=axStyle[0],
#       origin="lower", cmap='seismic', vmin=0, vmax=1, zorder=-1
#   )
  _, _, v = env.get_value(agent.Q_network, nx, ny)
  im = ax.imshow(
      v.T, interpolation='none', extent=axStyle[0], origin="lower",
      cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1
  )
  env.plot_trajectories(
      agent.Q_network, states=env.visual_initial_states, toEnd=True, ax=ax, q_func_1=agent.Q_decomposed_1, q_func_2=agent.Q_decomposed_2,
      c='k', lw=1.5
  )
  ax.set_title(f'{args.mode} Value & Rollout', fontsize=20)
  ax.set_xlabel(f'Name: {args.mode}/{fn}/Q-{restored_model_ix}', fontsize=8)
  
  if args.mode in ["RA", "R", "A"]:
    # = Action
    ax = axes[0]
    im = ax.imshow(
        actDistMtx.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap='seismic', vmin=0, vmax=actionNum - 1, zorder=-1
    )
    ax.set_title('Action', fontsize=20)
  
  else:
    # = Value
    ax = axes[0]
    if args.mode in ["RA", "R", "A"]:
        _, _, v = env.get_value(agent.Q_network, nx, ny)
    else:
        _, _, v = env.get_value(agent.Q_decomposed_1, nx, ny)
    im = ax.imshow(
        v.T, interpolation='none', extent=axStyle[0], origin="lower",
        cmap='seismic', vmin=vmin, vmax=vmax, zorder=-1

    ## FIXME WAS: ADD Q2 FOR RR PROBLEM
        
    )
    #   CS = ax.contour(
    #       xs, ys, v.T, levels=[0], colors='k', linewidths=2, linestyles='dashed'
    #   )
    
    env.plot_trajectories(
    agent.Q_decomposed_1, states=env.visual_initial_states, toEnd=True, ax=ax, q_func_1=agent.Q_decomposed_1, q_func_2=agent.Q_decomposed_2,
    c='k', lw=1.5
    )
    ax.set_title('A Value & Rollout', fontsize=20)
    load_tag = f'Load: {args.load_model_path_1.split("A/")[-1].split(".pth")[0]}'
    ax.set_xlabel(load_tag, fontsize=8)

  for axi, ax in enumerate(axes):
    env.plot_target_sets(ax=ax)
    # if axi == 0: env.plot_reach_avoid_set(ax=ax)
    env.plot_formatting(ax=ax)

#   fig.tight_layout()
  if storeFigure:
    if not args.skip_learn:
        figurePath = os.path.join(figureFolder, 'value_rollout_action.png')
    else:
        figurePath = os.path.join(figureFolder, 'value_rollout_action_replot.png')
    fig.savefig(figurePath)
  if plotFigure:
    plt.show()
    plt.pause(0.001)
  plt.close()
  
#   if not args.skip_learn:
#     trainDict['resultMtx'] = resultMtx
#     trainDict['actDistMtx'] = actDistMtx

if not args.skip_learn:
    save_obj(trainDict, filePath)
