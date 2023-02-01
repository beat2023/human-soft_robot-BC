# https://github.com/jerrylin1121/BCO/tree/tf2.0/models

import shutil
import tensorflow as tf
from keras import layers
from keras.models import load_model
from custom_env_MATLAB import *
# from stable_baselines3.common.env_checker import check_env
from sklearn.metrics import mean_absolute_error
import time
from utils import *
from copy import copy


# ######### Before running it, write on Matlab: matlab.engine.shareEngine ##############################################

########################################################################################################################


class Policy(tf.keras.Model):
    def __init__(self, action_shape):
        super(Policy, self).__init__()
        self.fc = tf.keras.Sequential([
            layers.Dense(128, activation='tanh'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='tanh'),
            layers.Dense(action_shape, activation='tanh')
        ])

    def call(self, states, training=None, mask=None):
        return self.fc(states)


class InverseDynamicsModel(tf.keras.Model):
    def __init__(self):
        super(InverseDynamicsModel, self).__init__()
        dir_name = '../IDM/IDM_MODEL'
        self.fc = load_model(dir_name)

    def call(self, features, training=None, mask=None):
        return self.fc(tf.reshape(features, [features.shape[0], 1, features.shape[1]]))


class BCO_all:
    def __init__(self, state_shape, action_shape, feature_shape, origin_pos):

        self.sampl_T = 0.2  # for an acquisition frequency of 5Hz

        self.state_dim = state_shape  # state dimension
        self.action_dim = action_shape  # action dimension
        self.feature_dim = feature_shape  # features dimension
        self.lr_policy = args.lr_policy  # model update learning rate for policy
        self.max_episodes = args.max_episodes  # maximum episode
        self.batch_percentage = args.batch_percentage  # batch size in percentage

        # build policy model and inverse dynamic model
        self.build_policy_model()
        self.build_idm_model()

        self.env = SoftRobotEnv(origin_pos)  # My environment

        # loss function and optimizer
        self.mse = tf.keras.losses.MeanSquaredError()
        self.optimizer_policy = tf.keras.optimizers.Adam(learning_rate=self.lr_policy)

        self.states_policy_all = []
        self.actions_policy_all = []

    def build_policy_model(self):
        """building the policy model as two fully connected layers with softsign"""
        self.policy = Policy(self.action_dim)
        self.policy_loss = tf.keras.metrics.Mean(name='policy_loss')

    def build_idm_model(self):
        """building the inverse dynamic model (LSTM)"""
        self.idm = InverseDynamicsModel()
        self.idm_loss = tf.keras.metrics.Mean(name='idm_loss')

    def eval_policy(self, states):
        """get the action by current state and past state"""
        states = np.array(states)
        # I eliminate proximal segments coordinates
        # states = np.delete(states, [0, 1, 2, 6, 7, 8])
        states_input = states
        flag = False
        if states.ndim == 1:
            flag = True
            states_input = np.reshape(states, (1, len(states)))
        action_out = self.policy(states_input)
        if flag:
            action_out = np.reshape(action_out, (action_out.shape[1],))
        return action_out

    def eval_idm(self, features):
        """get the action by inverse dynamic model from current set of features"""
        features = np.array(features)
        features_input = features
        flag = False
        if features.ndim == 1:
            flag = True
            features_input = np.reshape(features, (1, len(features)))
        action_out = self.idm(features_input)
        action_out = np.reshape(action_out, (action_out.shape[0], action_out.shape[2]))
        if flag:
            action_out = np.reshape(action_out, (action_out.shape[1],))
        return action_out

    def perform_policy(self, targets):
        """Perform the policy in the environment"""
        print('Waiting 3 seconds')
        sleep(3)
        States = []  # current states
        nStates = []  # states reached
        Actions = []
        prev_state = self.env.get_state()
        prev2_state = self.env.get_state()
        prev3_state = self.env.get_state()
        time_record = time.time()
        i = 0
        flag_stop = False
        flag_stuck = False
        stuck_ind = 0
        while not flag_stop:
            state = self.env.get_state()
            if i > 0:
                nStates.append(state)
            states = np.hstack((prev3_state, prev2_state, prev_state, state))
            action = self.eval_policy(states)  # get the action by current state
            action = check_action(action)
            States.append(state)
            Actions.append(action)
            self.env.step(action)
            # I stop post-demonstrations if the robot is stuck or if many iterations have been done
            if i > 10:
                diff_state = denormalize(state, 'state') - denormalize(prev_state, 'state')
                diff_state_2 = denormalize(prev_state, 'state') - denormalize(prev2_state, 'state')
                if sum(abs(diff_state)) < 5 and sum(abs(diff_state_2)) < 5:
                    stuck_ind += 1
                    if stuck_ind > 5:
                        flag_stuck = True
                else:
                    stuck_ind = 0
                if i > (len(targets) - 2) or flag_stuck:
                    flag_stop = True
            prev3_state = prev2_state
            prev2_state = prev_state
            prev_state = state

            time_to_sleep = self.sampl_T - (time.time() - time_record)
            if np.sign(time_to_sleep) != -1:
                sleep(time_to_sleep)
            time_record = time.time()
            i += 1

        nStates.append(self.env.get_state())  # To have the right number of reached states
        nStates = resampling(np.array(nStates), len(targets))

        self.env.reset()

        return States, nStates, Actions

    def demo_through_idm(self, S_in, reset):
        S = copy(S_in)
        # Initialize states and actions
        s_prev = self.env.get_state()
        s_prev2 = self.env.get_state()
        s_prev3 = self.env.get_state()
        A_2prev = normalize(self.env.action, 'action')  # initialize the previous-previous action
        A_prev = normalize(self.env.action, 'action')  # initialize the previous action
        A_tot = []
        States_policy = []
        states_idm = []
        for d in range(3):  # to use all the states of the trajectory
            S.append(S[-1])
        time_record = time.time()
        for d in range(len(S) - 3):
            s = self.env.get_state()
            if d > 0:
                states_idm.append(s)

            s_next = S[d + 1]
            s_next2 = S[d + 2]
            s_next3 = S[d + 3]

            features = np.hstack((s, s_prev2, s_prev, s_next, s_next2, s_next3, A_2prev, A_prev))
            states = np.hstack((s_prev3, s_prev2, s_prev, s))
            A = self.eval_idm(features)
            A = check_action(A)
            States_policy.append(states)
            A_tot.append(A.tolist())
            self.env.step(A)

            A_2prev = A_prev
            A_prev = np.reshape(A, (6,))
            s_prev3 = s_prev2
            s_prev2 = s_prev
            s_prev = s

            time_to_sleep = self.sampl_T - (time.time() - time_record)
            if np.sign(time_to_sleep) != -1:
                sleep(time_to_sleep)
            time_record = time.time()

        states_idm.append(self.env.get_state())  # to have the right number of states in the trajectory
        States_policy = np.array(States_policy)
        A_tot = np.array(A_tot)

        if reset:
            self.env.reset()

        return States_policy, states_idm, A_tot

    def reach_s_0(self, first):
        """Move the robot until it is close enough to the first position of the demo"""
        first = denormalize(np.array(first), 'state')
        current = self.env.agent_pos
        n_points = 15   # number of steps proportional to the distance
        add_points = np.zeros((n_points, 6))
        for i in range(6):
            add_points[:, i] = np.linspace(current[i], first[i], num=n_points)
        add_points = normalize(add_points, 'state')
        self.demo_through_idm(add_points.tolist(), reset=False)

    def post_demonstration(self, targets, s_0):
        """using policy to see the actual performance"""
        print('\nPOST-DEMONSTRATIONS')
        self.reach_s_0(s_0)
        outputs_idm = copy(self.outputs_best)
        States, nStates, Actions = self.perform_policy(targets)
        # Resample the states obtained in post-demo
        targets = denormalize(np.array(targets), 'state')
        outputs = denormalize(np.array(nStates), 'state')
        outputs_idm = denormalize(np.array(outputs_idm), 'state')
        # MAE of distal marker
        MAE = mean_absolute_error(targets[:, 3:], outputs[:, 3:])
        MAE_idm = mean_absolute_error(outputs_idm[:, 3:], outputs[:, 3:])
        Features = get_features(States, Actions)
        return np.array(Features), np.array(Actions), MAE, MAE_idm

    def eval_MAE_policy(self, numb):
        """getting the MAE score by current policy model"""
        print('\nEvaluating the MAE of the policy w.r.t. the average expert trajectory')
        targets = copy(self.targets_best)
        inputs = copy(self.inputs_best)
        output_idm = copy(self.outputs_best)
        self.reach_s_0(inputs[0])   # reach the initial state of demo
        _, nStates, _ = self.perform_policy(targets)
        # Resample the states obtained in testing
        targets = denormalize(np.array(targets), 'state')
        outputs = denormalize(np.array(nStates), 'state')
        # MAE of distal marker
        MAE = mean_absolute_error(targets[:, 3:], outputs[:, 3:])
        MAE_idm = mean_absolute_error(output_idm[:, 3:], outputs[:, 3:])
        save_test_policy(targets, output_idm, outputs, self.demo_mode, self.remap_type, self.demo_name, numb)
        return MAE, MAE_idm

    @tf.function  # Compiles a function into a callable TensorFlow graph (for better performance)
    def policy_train_step(self, states, action):
        """tensorflow 2.0 policy train step"""
        with tf.GradientTape() as tape:
            logits = self.policy(states)
            loss = self.mse(action, logits)
        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(grads, self.policy.trainable_variables))
        self.policy_loss(loss)

    def update_policy(self):
        """update policy model"""
        states = self.states_policy
        percent = self.batch_percentage
        # I eliminate proximal segments coordinates
        # states = np.delete(states, [0, 1, 2, 6, 7, 8], axis=1)
        states = list(states)
        actions = list(self.actions_policy)
        num = len(states)
        EPOCHS = 5
        for epoch in range(EPOCHS):
            idxs = get_shuffle_idx(num, percent)
            for idx in idxs:
                batch_s = tf.constant([states[i] for i in idx])
                batch_a = tf.gather(actions, idx)
                self.policy_train_step(batch_s, batch_a)

    def get_policy_loss(self):
        """get policy model loss"""
        loss = self.policy_loss.result()
        self.policy_loss.reset_states()
        return loss

    def train(self):
        """training the policy model and inverse dynamic model by behavioral cloning"""
        ckpt = tf.train.Checkpoint(model=self.policy)
        model_dir = 'BC_model/' + self.demo_mode + '/' + self.remap_type + '/' + self.demo_name
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        shutil.rmtree(model_dir)  # delete all files contained
        os.mkdir(model_dir)  # recreate the directory empty
        clear_post_demo(self.demo_mode, self.remap_type, self.demo_name)  # clear the post-demos of that type
        manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=10)

        demo_file_best = 'DEMOS_BEST/' + self.demo_mode + '/' + self.remap_type + '/transitions_shift_' + \
                         self.demo_name + '_best.txt'
        self.inputs_best, self.targets_best = load_demonstration(demo_file_best)

        print("\n[Training]")  # BCO uses the learned idm model to infer the missing actions
        print("\nUPDATING POLICY")
        MAE = np.zeros((self.max_episodes,))
        MAE_idm = np.zeros((self.max_episodes,))

        for i_episode in np.arange(self.max_episodes):
            print(f'\n\nStarting round number {i_episode}: best demonstration')
            S = copy(self.inputs_best)
            nS = copy(self.targets_best)
            self.reach_s_0(S[0])  # reach the initial state of demo
            States_policy, states_idm, A_tot = self.demo_through_idm(S, reset=True)
            self.outputs_best = states_idm

            # Finally, BCO uses the demonstration and the inferred actions to find a policy via behavioral cloning
            self.states_policy_all.append(States_policy)
            self.actions_policy_all.append(A_tot)
            if i_episode == 0:
                self.states_policy = States_policy
                self.actions_policy = A_tot
            else:  # I aggregate data with that of the previous episodes
                self.states_policy = np.vstack((self.states_policy, States_policy))
                self.actions_policy = np.vstack((self.actions_policy, A_tot))

            self.update_policy()
            policy_loss = self.get_policy_loss()
            print(f'Episode {i_episode}: policy loss {policy_loss}')

            # Extra interaction time used to learn a better model and improve its imitation policy
            F, A, MAE[i_episode], MAE_idm[i_episode] = self.post_demonstration(nS, S[0])
            print(f'MAE of post-demonstration number {i_episode}\n'
                  f'with original demo: {MAE[i_episode]}\n'
                  f'with idm-passed demo: {MAE_idm[i_episode]}')
            reached = F[1:, :6]
            reached = np.append(reached, np.reshape(reached[-1, :], (1, 6)), axis=0)
            save_post_demo(np.array(nS), states_idm, reached, self.demo_mode, self.remap_type, self.demo_name,
                           i_episode)

            print(f'MAE[i_episode]: {MAE[i_episode]}')
            print(f'min(MAE): {min(MAE[:i_episode + 1])}')
            print(f'MAE[i_episode] <= min(MAE)? {MAE[i_episode] <= min(MAE[:i_episode + 1])}')
            if MAE[i_episode] <= min(MAE[:i_episode + 1]):
                manager.save()  # saving model

            print('Waiting 3 seconds')
            sleep(3)

        print(f'All MAE values with original demo: {MAE.tolist()}')
        print(f'All MAE values with idm-passed demo: {MAE_idm.tolist()}')

    def test(self):
        test_n = 10    # number of times I want to test policy
        demo_best = 'DEMOS_BEST/' + self.demo_mode + '/' + self.remap_type + '/transitions_shift_' + \
                    self.demo_name + '_best.txt'
        self.inputs_best, self.targets_best = load_demonstration(demo_best)
        self.outputs_best = load_idm_perform_best(self.demo_mode, self.remap_type, self.demo_name)

        ckpt = tf.train.Checkpoint(model=self.policy)
        model_dir = 'BC_model/' + self.demo_mode + '/' + self.remap_type + '/' + self.demo_name
        ckpt.restore(tf.train.latest_checkpoint(model_dir))
        MAE_all = []
        MAE_idm_all = []
        print('\n[Testing]')
        for numb in range(test_n):
            MAE, MAE_idm = self.eval_MAE_policy(numb)
            MAE_all.append(MAE)
            MAE_idm_all.append(MAE_idm)
            print(f'\nRound number {numb}')
            print('MAE score with original demo: {:5.1f}'.format(MAE))
            print('MAE score with idm-passed demo: {:5.1f}'.format(MAE_idm))
        save_test_MAE(MAE_all, MAE_idm_all, self.demo_mode, self.remap_type, self.demo_name)

        MAE_avg = np.mean(MAE_all)
        MAE_idm_avg = np.mean(MAE_idm_all)
        print('\nAverage MAE score with original demo: {:5.1f}'.format(MAE_avg))
        print('Average MAE score with idm-passed demo: {:5.1f}'.format(MAE_idm_avg))

    def idm_perform(self):
        """Perform the demos in the environment to select the most easy to follow with the robot"""
        i_episode = 0
        MAE = []
        while True:
            demo_file = 'DEMOS_NORMALIZED/' + self.demo_mode + '/' + self.remap_type + '/transitions_shift_' + \
                        self.demo_name + '_' + str(i_episode) + '.txt'
            if not os.path.isfile(demo_file):
                # If the file doesn't exist I exit the loop
                break
            print(f'\n\nRound number {i_episode} of actuation in the environment')

            S, nS = load_demonstration(demo_file)
            self.reach_s_0(S[0])  # reach the initial state of demo
            _, reached, _ = self.demo_through_idm(S, reset=True)
            nS = np.array(nS)
            reached = np.array(reached)
            targets = denormalize(nS, 'state')
            outputs = denormalize(reached, 'state')
            MAE.append(mean_absolute_error(targets[:, 3:], outputs[:, 3:]))

            save_idm_perform(nS, reached, self.demo_mode, self.remap_type, self.demo_name, i_episode)

            print('Waiting 3 seconds')
            sleep(3)

            i_episode += 1

        print(f'MAE between original and idm-passed demo: {MAE}')
        self.best_episode_n = MAE.index(min(MAE))
        # Copy the best original demo
        demo_file_best = 'DEMOS_NORMALIZED/' + self.demo_mode + '/' + self.remap_type + '/transitions_shift_' + \
                         self.demo_name + '_' + str(self.best_episode_n) + '.txt'
        file_wname = 'DEMOS_BEST/' + self.demo_mode + '/' + self.remap_type + '/transitions_shift_' + \
                     self.demo_name + '_best.txt'
        shutil.copy(demo_file_best, file_wname)
        # Copy the best idm-passed demo
        idm_perform_best = 'IdmPerform/' + self.demo_mode + '/' + self.remap_type + '/' + \
                           self.demo_name + '_' + str(self.best_episode_n) + '.txt'
        file_wname_idm = 'IdmPerform/' + self.demo_mode + '/' + self.remap_type + '/' + \
                         self.demo_name + '_best.txt'
        shutil.copy(idm_perform_best, file_wname_idm)

        visualize_idm(self.demo_mode, self.remap_type, self.demo_name)

    def run(self):
        self.demo_name = args.demo_name
        self.remap_type = args.remap_type
        if args.mode == 'test':
            self.demo_mode = 'robot'
            self.test()
        if args.mode == 'test_human':
            self.demo_mode = 'human'
            self.test()
        if args.mode == 'train':
            self.demo_mode = 'robot'
            self.train()
        if args.mode == 'train_human':
            self.demo_mode = 'human'
            self.train()
        if args.mode == 'idm_perform':
            self.demo_mode = 'robot'
            self.idm_perform()
        if args.mode == 'idm_perform_human':
            self.demo_mode = 'human'
            self.idm_perform()


if __name__ == "__main__":
    origin = load_origin()
    bco = BCO_all(6, 6, 48, origin)  # 6 positions (state space), 6 pressures (action space) and 48 features
    bco.run()
    bco.env.close()
