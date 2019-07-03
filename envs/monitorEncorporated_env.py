from aa_simulation.contracts import *;
from rllab.envs.proxy_env import ProxyEnv;
from aa_simulation.envs.base_env import VehicleEnv;
from rllab.policies.base import Policy; 
import numpy as np;

from rllab.core.serializable import Serializable;
from rllab.spaces import Box;

from rllab.envs.base import Step;

def isProperMonitorSubFormula(thisProposedSubFormula):
    return isinstance(thisProposedSubFormula, str);

# In triple quotes below: the original implementation of 
# isProperMonitorSubFormula prior to running into 
# substantial problems pickling functions between components
"""
super(MonitorEncorporatedEnv, self).__init__(wrapped_env);
    if(str(type(thisProposedSubFormula))  != "<class 'function'>"):
        return False;
    if(thisProposedSubFormula.__code__.co_argcount != 2):
        return False;
    return True;
"""


class MonitorEncorporatedEnv(ProxyEnv):
    """
    MonitorEncorporatedEnv: this class provides a way for transforming an instance of the
    VehicleEnv class (the "wrapped environment") to an environment where the monitor is used
    in a variety of ways. The monitor information is provided as a list of the quantitative
    subformulas - for instance if the monitor is ((A < B) AND (C < D) AND (E < F)), then the 
    monitor is provided as [B -A, D -C, E -F]. The monitor information can be used in any subset
    of the following:
        (1) Activate fallback controller: in the case the monitor is violated, a fallback controller
            is used to dictate the actions to be take as opposed to the agent interacting with 
            the environment. This occurs in all and only the situations where the monitor is 
            violated. To disable this functionality, simply pass in None for codeForFallbackController;
            the actions provided through the step-function (see the code below) to the environment 
            then will always be acted out.
        (2) Influence the reward returned; the reward returned by the environment is a weighted
            combination of the reward given by the wrapped-environment and the value given by the 
            quantitative monitor. Specifically, the reward given is:
                reward = rewardFromWrappedEnvironment + \
                        weightForQuantMonitorValueInReward * min(B -A, D -C, E -F)
            To effectively disable this functionality, set weightForQuantMonitorValueInReward to 0.0 .
        (3) Additional features in observations: in addition to the features provided by the 
            wrapped-environment, the quantitative-monitor subformulas can be provided as additional
            features for a state. For instance, if the initial feature vector is :
                 [f_1, f_2, ..., f_{n-1}, f_n]
            the features can be expanded to include:
                 [f_1, f_2, ..., f_{n-1}, f_n, B -A, D -C, E -F]
            Note that since we use the quantitative monitor subformulas, the features vary over the
            set of the state-space where the monitor is not violated. This is in contrast to if
            the subformulas from the original monitor, in which case the binary values would not
            vary of the safe-set - the moment any one of them changes, the environment which trigger
            the fallback controller to kind the vehicle, which makes having those features in such
            an arrangement have little utility. To enable these additional features, set 
            useQuantMonitorSubformulasAsFeatures to true, and to disable, set 
            useQuantMonitorSubformulasAsFeatures to false.
    Again, any subset of the above three options is valid - so there are at least 8 general modes of 
    operation for this class.

    
    A Note on Some Unfortunate Hacks Made To Get The rllab Infrastructure to Work With This Code:
        Unfortunately, various parts of the rllab code try to do clever things with pickling is
        saving results and passing parameters around in the infrastructure. This limits how much plain
        functions can be passed around as parameters - while cloudpickle can be substituted in 
        many places for pickle in the rllab code, at least three challenges remain there: (1) rllab uses
        some functionality of pickle not supported by cloudpickle (specific attributes pickle has that
        cloudpickle does not), (2) rllab is a project outside the general control of the aa-group, and
        the code base for it has been frozen for some time in favor of developing a new platform; as
        such, we would have to modify our own local copy of rllab and distribute to any in the aa-group
        who want to use it, (3) in addition to the python package "pickle", rllab also takes advantage of
        numpy pickle functions that apparently have some similar issues. 

        As a work-around to the difficulties listed above, the code was change to use code for functions
        in place of python implementations of the functions. That is, instead of passing in, say,
            lambda x: x +2 
        the code requires that the string
            "x + 2"
        be passed in. Specifically, the elements of quantitativeMonitorSubFormulas must be strings that can
        be evaluated by the python built-in eval ,  and codeForFallbackController must be text 
        evaluatable by the python built-in exec and must define the function fallbackController .
        Plans for near-future development include investigating better ways to handle the circumstances.
        For the first swing at developing these functionalities, this arrangement should be sufficient
        and not overly brittle nor overly complex.
    """

    def __init__(self, wrapped_env, quantitativeMonitorSubFormulas, \
            weightForQuantMonitorValueInReward, codeForFallbackController, useQuantMonitorSubformulasAsFeatures):
        requires(isinstance(wrapped_env, VehicleEnv));
        requires(isinstance(quantitativeMonitorSubFormulas, list)); 
        # NOTE: we allow quantitativeMonitorSubFormulas to be an empty list, 
        #     in which case no monitor violations should ever occur
        requires(all([isProperMonitorSubFormula(x) for x in quantitativeMonitorSubFormulas])); 
        requires(isinstance(weightForQuantMonitorValueInReward, float));
        # NOTE: we allow weightForQuantMonitorValueInReward to be negative, in 
        #     case the agent would be rewarded for violating the monitor condition.
        #     This might be useful for testing or to empirically judge the 
        #     influence of the monitor signal encorporated via the reward function.
        requires(codeForFallbackController == None or isinstance(codeForFallbackController, str));
        requires(isinstance(useQuantMonitorSubformulasAsFeatures, bool));

        # NOTE: we cannot do
        #         ProxyEnv.__init__(self, wrapped_env);
        #     or
        #         super(MonitorEncorporatedEnv, self).__init__(wrapped_env);  
        #     since the init function here (unlike the ProxyEnv class) takes in multiple
        #     arguments and results in local() not being able to find all of them if
        #     we try calling as listed above.
        Serializable.quick_init(self, locals())
        self._wrapped_env = wrapped_env

        # TODO: consider including python-ic leading underscore as necessary...
        self.quantitativeMonitorSubFormulas = quantitativeMonitorSubFormulas;
        self.weightForQuantMonitorValueInReward = weightForQuantMonitorValueInReward;
        if(codeForFallbackController != None):
            exec(codeForFallbackController);
            self.fallbackController = locals()["fallbackController"];
        else:
            self.fallbackController = None;

        self.useQuantMonitorSubformulasAsFeatures = useQuantMonitorSubformulasAsFeatures; 

        # TODO: Select better informed values of self._action for prior to
        #     the when the controller makes it first decision Grep over this
        #     file to see where self._action is used and why the value prior to
        #     the first choice might have some impact.
        self._action = np.array([0,0]); # Note that this is the actual action performed on the 
            # environment, not necessarly the same as self._wrapped_env.action -
            # in the case of a monitor violation, and if a fallback controller is
            # specified, then the action is dictated by the fallback-controller, not
            # the initial policy.

        # NOTE: Setting the two _state variable below is important for calculating
        #     the acceleration fed into the quantitative-monitor subformulas. See
        #     the function evaluate_quantitativeMonitorSubFormulas .
        self._state = np.zeros(self.observation_space.flat_dim,)

        return;


    def getAxiluraryInformation(self, state):
        fakeTime = 0.0; #the time is not actually used in the dynamics in question...
        # Note below we use self._action not self._wrapped_env.action, since we want the
        # actual action performed, not the one the wrapped-controller would have done....
        state_dot = self._wrapped_env._model._dynamics(state, fakeTime, self._action);
        """
        recall:
        state_dot[0] = pos_x_dot
        state_dot[1] = pos_y_dot
        state_dot[2] = yaw_rate
        state_dot[3] = v_x_dot
        state_dot[4] = v_y_dot
        state_dot[5] = yaw_rate_dot
        """
        return state_dot[3:6]; # returning the accelerations.


    def evaluate_quantitativeMonitorSubFormulas(self, state, action):
        axiluraryInformation = self.getAxiluraryInformation(state)
        listToReturn = [\
            eval(x, {"state" : state, "axiluraryInformation" :  axiluraryInformation}) \
            for  x in self.quantitativeMonitorSubFormulas];
        ensures(isinstance(listToReturn, list));
        ensures(len(listToReturn) == len(self.quantitativeMonitorSubFormulas));
        return listToReturn;

    def getMin_evaluate_quantitativeMonitorSubFormulas(self, state, action):
        # This function handles the edge case where self.quantitativeMonitorSubFormulas is an 
        # empty list - helps avoid silly errors that might result from the more 
        # straight-forward use of min(self.evaluate_quantitativeMonitorSubFormulas(state, action))
        # at various locations.
        if(len(self.quantitativeMonitorSubFormulas) == 0):
            return 0.0; # NOTE: we consider the monitor to be violated when the value from
                # the quantitative monitor is negative, so returning zero should not consistute
                # a monitor violation.
        else:
            return min(self.evaluate_quantitativeMonitorSubFormulas(state, action));
        raise Exception("Control should never reach here");
        return;


    def reset(self):
        """
        Reset environment back to original state.
        """
        self._action = np.array([0,0]); # None
        self._wrapped_env._state =  self._action;
        self._state = self._wrapped_env.get_initial_state
        self._wrapped_env._state = self._state;
        observation = self.state_to_observation(self._state)

        # Reset renderer if available
        if self._wrapped_env._renderer is not None:
            self._wrapped_env._renderer.reset()

        return observation


    def helper_step(self, action):
        """
        Move one iteration forward in simulation.
        """
        if action[0] < 0:   # Only allow forward direction
            action[0] = 0
        nextstate = self._wrapped_env._model.state_transition(self._state, action,
                self._wrapped_env._dt)
        self._state = nextstate
        # Notice below that we use the
        # state_to_observation and get_reward functions defined in this class as oppossed to the
        # ones defined in the self._wrapped_class, hence the need to reimplement this
        # function (helper_step) as oppossed to simply calling self._wrapped_class.step 
        reward, info = self.get_reward(nextstate, action)
        observation = self.state_to_observation(nextstate)
        return Step(observation=observation, reward=reward, done=False,
                dist=info['dist'], vel=info['vel'], kappa=self._wrapped_env._model.kappa)


    def step(self, action):
        if(self.fallbackController != None): 
            monitorHasBeenViolated = (\
                self.getMin_evaluate_quantitativeMonitorSubFormulas(self._wrapped_env._state, action) < 0.0 );
            action = self.fallbackController(self._wrapped_env._state);  
        self._action = action;
        # TODO: consider whether we should also set self._wrapped_env._action or make
        #     the opion of whether or not to do that a variable passed in to the 
        #     init function of this class.
        return self.helper_step(action);


    def get_reward(self, state, action):
        reward ,info = self._wrapped_env.get_reward(state, action); 
        if(self.weightForQuantMonitorValueInReward != 0.0): # this conditional prevents unnecessary
            # computation, but is not strictly needed.
            minimumQuantMonitorValue = self.getMin_evaluate_quantitativeMonitorSubFormulas(state, action);
            reward = reward + self.weightForQuantMonitorValueInReward * minimumQuantMonitorValue;
        return reward, info;


    def state_to_observation(self, state):
        originalObs = self._wrapped_env.state_to_observation(state);
        if(self.useQuantMonitorSubformulasAsFeatures):
            quantMonitorInput = np.array(self.evaluate_quantitativeMonitorSubFormulas(state, self._action));
            originalObs = np.concatenate([originalObs, quantMonitorInput]);
        return originalObs;


    @property
    def observation_space(self):
        """
        Define the shape of input vector to the neural network.
        """
        if(not self.useQuantMonitorSubformulasAsFeatures):
            return Box(low=-np.inf, high=np.inf, shape=(5,));
        else:
            return Box(low=-np.inf, high=np.inf, shape=(5 +len(self.quantitativeMonitorSubFormulas)));
        raise Exception("Control should never reach here");
        return;


    @property
    def get_initial_state(self):
        state = self._wrapped_env.get_initial_state; 
        # NOTE: Setting the two state variables below are important for calculating
        #     the acceleration fed into the quantitative-monitor subformulas. See
        #     the function evaluate_quantitativeMonitorSubFormulas
        self._state = state;
        self._wrapped_env._state = state;
        return state


    def get_action(observation):
        return self._wrapped_env.get_action(observation);


