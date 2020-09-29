import numpy as np
from derby.core.basic_structures import AuctionItemSpecification
from derby.core.ad_structures import Campaign
from derby.core.auctions import KthPriceAuction
from derby.core.pmfs import PMF
from derby.core.environments import train, generate_trajectories, OneCampaignNDaysEnv
from derby.core.agents import Agent
from derby.core.policies import *
from pprint import pprint
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import sys
import os
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class Experiment:

    def __init__(self):
        self.auction_item_specs = [
                        AuctionItemSpecification(name="male", item_type={"male"}),
        ]
        self.campaigns = [
                        Campaign(10, 100, self.auction_item_specs[0]),
        ]
        self.first_price_auction = KthPriceAuction(1)
        self.second_price_auction = KthPriceAuction(2)


    def setup_1(self, height_tooth1, midpoint, height_tooth2, maxpoint, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 1
        })
        auction_item_spec_ids = [spec.uid for spec in auction_item_spec_pmf.items()]
        campaign_pmf = PMF({
                    self.campaigns[0] : 1
        })
        if debug:
            print("campaigns: {}".format(campaign_pmf.items()))

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
        reward_func = OneCampaignNDaysEnv.sawtooth_reward_func(height_tooth1=height_tooth1, midpoint=midpoint, 
                                                               height_tooth2=height_tooth2, maxpoint=maxpoint)
        env.set_reward_func(reward_func)
        print("setup: setup_1")
        print("height_tooth1: {}, midpoint: {}, height_tooth2: {}, maxpoint: {}".format(
                        height_tooth1, midpoint, height_tooth2, maxpoint))
        return env, auction_item_spec_ids


    def get_states_scaler_descaler(self, samples): 
        '''
        :param samples: an array of shape [num_of_samples, num_of_agents, state_size]
        containing samples of states of the environment.
        '''
        samples_shape = samples.shape
        # reshape to [num_of_samples * num_of_agents, state_size]
        samples = samples.reshape(-1, samples_shape[-1])
        states_scaler = MinMaxScaler()
        states_scaler.fit(samples)

        def scale_states_func(states):
            # Input states is an array of shape [batch_size, episode_length, folded_state_size]
            # reshape to [batch_size * episode_length * fold_size, state_size].
            states_reshp = states.reshape(-1, samples_shape[-1])
            # Scale the states.
            scaled_states = states_scaler.transform(states_reshp)
            # Reshape back to original shape.
            return scaled_states.reshape(states.shape)

        def descale_states_func(states):
            # Input states is an array of shape [batch_size, episode_length, folded_state_size]
            # reshape to [batch_size * episode_length * fold_size, state_size].
            states_reshp = states.reshape(-1, samples_shape[-1])
            # Scale the states.
            descaled_states = states_scaler.inverse_transform(states_reshp)
            # Reshape back to original shape.
            return descaled_states.reshape(states.shape)

        return states_scaler, scale_states_func, descale_states_func


    def get_actions_scaler_descaler(self, samples):
        '''
        :param samples: an array of shape [num_of_samples, num_of_agents, state_size]
        containing samples of states of the environment.
        '''
        samples_shape = samples.shape
        # reshape to [num_of_samples * num_of_agents, state_size]
        samples = samples.reshape(-1, samples_shape[-1])
        # an array of shape [num_of_samples * num_of_agents, 1], containing the sample budgets.
        budget_samples = samples[:,1:2]
        actions_scaler = MinMaxScaler()
        actions_scaler.fit(budget_samples)

        def descale_actions_func(scaled_actions):
            # Assuming scaled_actions is of shape [batch_size, episode_length, num_of_subactions, subactions_size],
            # where subactions_size is bid_vector_size (i.e. vector [auction_item_spec_id, bid_per_item, total_limit]).
            # Slice out the 1st field of the vector (i.e. the auction_item_spec_id field).
            sa_without_ais = scaled_actions[:,:,:,1:]
            # Reshaping from [batch_size, episode_length, num_of_subactions, subactions_size-1]
            # to [batch_size * episode_length * num_of_subactions, subactions_size-1]
            sa_without_ais_reshp = sa_without_ais.reshape(-1, sa_without_ais.shape[-1])
            # Descale the remaining fields of the bid vectors (i.e. [bid_per_item, total_limit]).
            descaled_actions_without_ais = actions_scaler.inverse_transform(sa_without_ais_reshp)
            # Reshape back to [batch_size, episode_length, num_of_subactions, subactions_size-1]
            descaled_actions_without_ais = descaled_actions_without_ais.reshape(sa_without_ais.shape)
            # Concatenate the sliced out 1st fields with the descaled other fields.
            descaled_actions = np.concatenate((scaled_actions[:,:,:,0:1], descaled_actions_without_ais), axis=3)
            return descaled_actions

        def scale_actions_func(descaled_actions):
            # Assuming scaled_actions is of shape [batch_size, episode_length, num_of_subactions, subactions_size],
            # where subactions_size is bid_vector_size (i.e. vector [auction_item_spec_id, bid_per_item, total_limit]).
            # Slice out the 1st field of the vector (i.e. the auction_item_spec_id field).
            da_without_ais = descaled_actions[:,:,:,1:]
            # Reshaping from [batch_size, episode_length, num_of_subactions, subactions_size-1]
            # to [batch_size * episode_length * num_of_subactions, subactions_size-1]
            da_without_ais_reshp = da_without_ais.reshape(-1, da_without_ais.shape[-1])
            # Scale the remaining fields of the bid vectors (i.e. [bid_per_item, total_limit]).
            scaled_actions_without_ais = actions_scaler.transform(da_without_ais_reshp)
            # Reshape back to [batch_size, episode_length, num_of_subactions, subactions_size-1]
            scaled_actions_without_ais = scaled_actions_without_ais.reshape(da_without_ais.shape)
            # Concatenate the sliced out 1st fields with the scaled other fields.
            scaled_actions = np.concatenate((descaled_actions[:,:,:,0:1], scaled_actions_without_ais), axis=3)
            return scaled_actions

        return actions_scaler, scale_actions_func, descale_actions_func


    def get_transformed(self, env):
        # An array of shape [num_of_samples, num_of_agents, state_size].
        # If agents have not been set, num_of_agents defaults to 1.
        samples = env.get_states_samples(10000)
        _, scale_states_func, _  = self.get_states_scaler_descaler(samples)
        actions_scaler, scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(samples)
        
        # shape (num_of_nonzero_samples, state_size)
        nonzero_samples = tf.boolean_mask(samples, tf.math.logical_not(tf.math.logical_and(samples[:,:,0] == 0, samples[:,:,1] == 0)))
        avg_budget_per_reach = tf.reduce_mean(nonzero_samples[:,1] / nonzero_samples[:,0])
        scaled_avg_bpr = actions_scaler.transform([[avg_budget_per_reach]])[0][0]
        return scale_states_func, actions_scaler, scale_actions_func, descale_actions_func, scaled_avg_bpr


    def run(self, env, agents, num_days, num_trajs, num_epochs, horizon_cutoff, vectorize=True, debug=False):
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = horizon_cutoff
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = vectorize
        env.init(agents, num_of_days)
        print("agent policies: {}".format([agent.policy for agent in env.agents]))

        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff, debug=debug)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))

            if ((i+1) % 50) == 0:
                avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                max_last_50_epochs = [(agent.name, np.max(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
                print("Max of last 50 epochs: {}".format(max_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))


    def exp_1(self, num_days, num_trajs, num_epochs, lr, 
                    a, midpoint, b, maxpoint, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1(a, midpoint, b, maxpoint)
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v6_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2(self, num_days, num_trajs, num_epochs, lr, 
                    a, midpoint, b, maxpoint, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1(a, midpoint, b, maxpoint)
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3(self, num_days, num_trajs, num_epochs, lr, 
                    a, midpoint, b, maxpoint, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1(a, midpoint, b, maxpoint)
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        scaled_maxpoint = actions_scaler.transform([[maxpoint]])[0][0]

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_AdaptiveLR_v6_1_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_maxpoint, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4(self, num_days, num_trajs, num_epochs, lr, 
                    a, midpoint, b, maxpoint, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1(a, midpoint, b, maxpoint)
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        scaled_maxpoint = actions_scaler.transform([[maxpoint]])[0][0]

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_AdaptiveLR_v3_2_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_maxpoint, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


if __name__ == '__main__':
    exp = sys.argv[1]
    num_days = int(sys.argv[2])
    num_trajs = int(sys.argv[3])
    num_epochs = int(sys.argv[4])
    lr = float(sys.argv[5])
    a = float(sys.argv[6])
    midpoint = float(sys.argv[7])
    b = float(sys.argv[8])
    maxpoint = float(sys.argv[9])

    try:
        debug_str = sys.argv[6].strip().lower()
        if (debug_str == 't') or (debug_str == 'true'):
            debug = True
        else:
            debug = False
    except:
        debug = False
    experiment = Experiment()
    function_mappings = {
        'exp_1': experiment.exp_1, # setup 1, REINFORCE_Baseline_Gaussian_v6_1 vs. None
        'exp_2': experiment.exp_2, # setup 1, REINFORCE_Baseline_Gaussian_v3_2 vs. None
        'exp_3': experiment.exp_3, # setup 1, REINFORCE_Baseline_AdaptiveLR_v6_1_Gaussian vs. None
        'exp_4': experiment.exp_4, # setup 1, REINFORCE_Baseline_AdaptiveLR_v3_2_Gaussian vs. None
    }
    try:
        exp_func = function_mappings[exp]
    except KeyError:
        raise ValueError('invalid input')

    print("Running experiment {}".format(exp_func.__name__))
    states, actions, rewards = exp_func(num_days, num_trajs, num_epochs, lr, 
                                        a, midpoint, b, maxpoint, debug=debug)
    if debug:
        if states is not None:
            print("states shape: {}".format(states.shape))
            print("states:\n{}".format(states))
            print()
        if actions is not None:
            print("actions:\n{}".format(actions))
            print()
        if rewards is not None:
            print("rewards:\n{}".format(rewards))