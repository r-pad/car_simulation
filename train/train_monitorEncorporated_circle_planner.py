#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: DBayani
largely based on train_straight_planner.py by edwardahn 

Demonstrations of training with MonitorEncorporatedEnv
environment wrapped around the straight vehicle environment.
"""

import argparse

import joblib
import lasagne.init as LI
import lasagne.layers as L
import lasagne.nonlinearities as LN
import numpy as np

from rllab.algos.trpo import TRPO
from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
# from rllab.envs.base import Env
from rllab.misc import ext, logger
from rllab.misc.instrument import run_experiment_lite, VariantGenerator
from rllab.misc.resolve import load_class
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.cpo.baselines.linear_feature_baseline import LinearFeatureBaseline

from aa_simulation.envs.circle.circle_env import CircleEnv
from aa_simulation.envs.monitorEncorporated_env import MonitorEncorporatedEnv; 

# Pre-trained policy and baseline
policy = None
baseline = None



from rllab.policies.base import Policy;


# We keep in the below dead-code as a reminder that the ideal way the 
# fallback-controller would be specified is with an instance of the policy class.
"""
class DummbyFallbackPolicy(Policy):
    def get_action(self, observation):
        return np.array([0,0]);
"""



def run_task(vv, log_dir=None, exp_name=None):
    global policy
    global baseline

    trpo_stepsize = 0.01
    trpo_subsample_factor = 0.2

    # Check if variant is available
    if vv['model_type'] not in ['BrushTireModel', 'LinearTireModel']:
        raise ValueError('Unrecognized model type for simulating robot')
    if vv['robot_type'] not in ['MRZR', 'RCCar']:
        raise ValueError('Unrecognized robot type')


    # Load environment
    baseEnv = CircleEnv(
        target_velocity=vv['target_velocity'],
        radius=vv['radius'],
        dt=vv['dt'],
        model_type=vv['model_type'],
        robot_type=vv['robot_type']
    )
    env = MonitorEncorporatedEnv(\
        wrapped_env = baseEnv, \
        quantitativeMonitorSubFormulas = vv["quantitativeMonitorSubFormulas"], \
        weightForQuantMonitorValueInReward = vv["weightForQuantMonitorValueInReward"], \
        codeForFallbackController = vv["codeForFallbackController"], \
        useQuantMonitorSubformulasAsFeatures = vv["useQuantMonitorSubformulasAsFeatures"]\
        );

    # Save variant information for comparison plots
    variant_file = logger.get_snapshot_dir() + '/variant.json'
    logger.log_variant(variant_file, vv)

    # Set variance for each action component separately for exploration
    # Note: We set the variance manually because we are not scaling our
    #       action space during training.
    init_std_speed = vv['target_velocity'] / 4
    init_std_steer = np.pi / 6
    init_std = [init_std_speed, init_std_steer]

    # Build policy and baseline networks
    # Note: Mean of policy network set to analytically computed values for
    #       faster training (rough estimates for RL to fine-tune).
    if policy is None or baseline is None:
        wheelbase = 0.257
        target_velocity = vv['target_velocity']
        target_steering = np.arctan(wheelbase / vv['radius'])  # CCW
        output_mean = np.array([target_velocity, target_steering])
        hidden_sizes = (32, 32)

        # In mean network, allow output b values to dominate final output
        # value by constraining the magnitude of the output W matrix. This is
        # to allow faster learning. These numbers are arbitrarily chosen.
        W_gain = min(vv['target_velocity'] / 5, np.pi / 15)

        mean_network = MLP(
            input_shape=(env.spec.observation_space.flat_dim,),
            output_dim=env.spec.action_space.flat_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=LN.tanh,
            output_nonlinearity=None,
            output_W_init=LI.GlorotUniform(gain=W_gain),
            output_b_init=output_mean
        )
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=hidden_sizes,
            init_std=init_std,
            mean_network=mean_network
        )
        baseline = LinearFeatureBaseline(
            env_spec=env.spec,
            target_key='returns'
        )

    # Reset variance to re-enable exploration when using pre-trained networks
    else:
        policy._l_log_std = ParamLayer(
            policy._mean_network.input_layer,
            num_units=env.spec.action_space.flat_dim,
            param=LI.Constant(np.log(init_std)),
            name='output_log_std',
            trainable=True
        )
        obs_var = policy._mean_network.input_layer.input_var
        mean_var, log_std_var = L.get_output([policy._l_mean, policy._l_log_std])
        policy._log_std_var = log_std_var
        LasagnePowered.__init__(policy, [policy._l_mean, policy._l_log_std])
        policy._f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var]
        )

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=600,
        max_path_length=env.horizon,
        n_itr=600,
        discount=0.99,
        step_size=trpo_stepsize,
        plot=False,
    )
    algo.train()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str,
            help='Path to snapshot file of pre-trained network')
    args = parser.parse_args()
    return args


def main():
    global policy
    global baseline

    # Load pre-trained network if available
    args = parse_arguments()
    if args.network is not None:
        data = joblib.load(args.network)
        policy = data['policy']
        baseline = data['baseline']
        use_pretrained = True
    else:
        use_pretrained = False

    # Run multiple experiment variants at once
    vg = VariantGenerator()

    # Non-configurable parameters (do not change)
    vg.add('trajectory', ['Circle'])
    vg.add('objective', ['TargetVelocity'])
    vg.add('algo', ['TRPO'])


    targetVelocity = 1.0
    epsilonValue = 0.05;

    # Configurable parameters
    #   Options for model_type: 'BrushTireModel', 'LinearTireModel'
    #   Options for robot_type: 'MRZR', 'RCCar'
    seeds = [100, 200]
    robot_type = 'RCCar'
    use_ros = False
    vg.add('seed', seeds)
    vg.add('target_velocity', [targetVelocity])
    vg.add('radius', [1.0])
    vg.add('dt', [0.1])
    vg.add('eps', [epsilonValue])
    vg.add('model_type', ['BrushTireModel'])
    vg.add('robot_type', [robot_type])
    # We are mostly uninterested in the negative values below, since they would encourage
    #     violating the monitor....
    vg.add('weightForQuantMonitorValueInReward', [0.0, 2.0, 0.125, -0.125, 1.0, 0.25, 0.5, -2.0]);
    vg.add('useQuantMonitorSubformulasAsFeatures', [True, False]);

    quantMonitorSubformula_carDrivingAwayFromLine = str(epsilonValue) + "- state[0]"; # I.e., if more than 
        # epsilon away from the target raduis, then the quantitative monitor is violated.
    quantitativeMonitorSubFormulas = \
        [quantMonitorSubformula_carDrivingAwayFromLine];

    codeForFallbackController = """
import numpy as np;
def fallbackController(observation):
    deltaRaduis = observation[0]

    amountToSteerOffCenter = 0.1; # very slight to avoid oversteering... but completely
        # fails to consider drifting. As such, this is VERY MUCH a first-swing attempt
        # with a fallback controller that we would not actually use.
    velocityToUse = """ + str(targetVelocity) + """;

    # steers toward the line
    steeringAngle = amountToSteerOffCenter * np.sign(deltaRaduis); # If we are outside the circle, we 
        # stear toward the inside, if we are on the inside, we steer toward the outside....
    return np.array([steeringAngle, velocityToUse]);
"""; 

    vg.add("codeForFallbackController", [codeForFallbackController, None]);
    vg.add("quantitativeMonitorSubFormulas", [quantitativeMonitorSubFormulas, []]);

    print('Number of Configurations: ', len(vg.variants()))

    # Run each experiment variant
    for vv in vg.variants():
        run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            n_parallel=1,
            snapshot_mode='last',
            seed=vv['seed']
        )


if __name__ == '__main__':
    main()

