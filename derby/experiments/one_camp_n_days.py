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
                        AuctionItemSpecification(name="female", item_type={"female"}),
                        AuctionItemSpecification(name="young", item_type={"young"}),
                        AuctionItemSpecification(name="old", item_type={"old"}),
                        AuctionItemSpecification(name="rich", item_type={"rich"}),
                        AuctionItemSpecification(name="poor", item_type={"poor"}),
                        AuctionItemSpecification(name="male young", item_type={"male", "young"}),
                        AuctionItemSpecification(name="male old", item_type={"male", "old"}),
                        AuctionItemSpecification(name="female young", item_type={"female", "young"}),
                        AuctionItemSpecification(name="female old", item_type={"female", "old"}),
                        AuctionItemSpecification(name="male rich", item_type={"male", "rich"}),
                        AuctionItemSpecification(name="male poor", item_type={"male", "poor"}),
                        AuctionItemSpecification(name="female rich", item_type={"female", "rich"}),
                        AuctionItemSpecification(name="female poor", item_type={"female", "poor"})
        ]
        self.campaigns = [
                        Campaign(10, 100, self.auction_item_specs[0]),
                        Campaign(10, 100, self.auction_item_specs[1]),
                        Campaign(20, 200, self.auction_item_specs[0]),
                        Campaign(20, 200, self.auction_item_specs[1]),
                        Campaign(30, 250, self.auction_item_specs[0]),
                        Campaign(30, 250, self.auction_item_specs[1]),
                        Campaign(10, 200, self.auction_item_specs[2]),
                        Campaign(10, 150, self.auction_item_specs[3]),
                        Campaign(20, 400, self.auction_item_specs[2]),
                        Campaign(20, 300, self.auction_item_specs[3]),
        ]
        self.first_price_auction = KthPriceAuction(1)
        self.second_price_auction = KthPriceAuction(2)


    def setup_1(self, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [spec.uid for spec in auction_item_spec_pmf.items()]
        campaign_pmf = PMF({
                    self.campaigns[1] : 1
        })
        if debug:
            print("campaigns: {}".format(campaign_pmf.items()))

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
        print("setup: setup_1")
        return env, auction_item_spec_ids


    def setup_2(self, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 1,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [spec.uid for spec in auction_item_spec_pmf.items()]
        campaign_pmf = PMF({
                    self.campaigns[0] : 1,
                    self.campaigns[1] : 1
        })
        if debug:
            print("campaigns: {}".format(campaign_pmf.items()))

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
        print("setup: setup_2")
        return env, auction_item_spec_ids


    def setup_3(self, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 1,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [spec.uid for spec in auction_item_spec_pmf.items()]
        campaign_pmf = PMF({
                    self.campaigns[0] : 1,
                    self.campaigns[1] : 1
        })
        if debug:
            print("campaigns: {}".format(campaign_pmf.items()))

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 4
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
        print("setup: setup_3")
        return env, auction_item_spec_ids


    def setup_4(self, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 1,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [spec.uid for spec in auction_item_spec_pmf.items()]
        campaign_pmf = PMF({
                    self.campaigns[0] : 1,
                    self.campaigns[1] : 1
        })
        if debug:
            print("campaigns: {}".format(campaign_pmf.items()))

        num_items_per_timestep_min = 10
        num_items_per_timestep_max = 21
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
        print("setup: setup_4")
        return env, auction_item_spec_ids


    def setup_5(self, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 1,
                    self.auction_item_specs[1] : 1,
                    self.auction_item_specs[2] : 1,
                    self.auction_item_specs[3] : 1
        })
        auction_item_spec_ids = [spec.uid for spec in auction_item_spec_pmf.items()]
        campaign_pmf = PMF({
                    self.campaigns[0] : 1,
                    self.campaigns[1] : 1,
                    self.campaigns[6] : 1,
                    self.campaigns[7] : 1
        })
        if debug:
            print("campaigns: {}".format(campaign_pmf.items()))

        num_items_per_timestep_min = 4
        num_items_per_timestep_max = 9
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
        print("setup: setup_5")
        return env, auction_item_spec_ids


    def setup_6(self, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 1,
                    self.auction_item_specs[1] : 1,
        })
        auction_item_spec_ids = [spec.uid for spec in auction_item_spec_pmf.items()]
        campaign_pmf = PMF({
                    self.campaigns[0] : 3,
                    self.campaigns[1] : 3,
                    self.campaigns[2] : 2,
                    self.campaigns[3] : 2,
                    self.campaigns[4] : 1,
                    self.campaigns[5] : 1,
        })
        if debug:
            print("campaigns: {}".format(campaign_pmf.items()))

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 3
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
        print("setup: setup_6")
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

    def exp_1(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = self.auction_item_spec_pmf
        campaign_pmf = self.campaign_pmf
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1000
        num_items_per_timestep_max = 1001
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)

        agents = [
                    Agent("agent1", FixedBidPolicy(1.0, 1.0, auction_item_spec=auction_item_specs[0])), 
                    Agent("agent2", FixedBidPolicy(2.0, 2.0, auction_item_spec=auction_item_specs[1]))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        # Vectorize is True
        env.vectorize = True
        env.init(agents, num_of_days)
        states, actions, rewards = generate_trajectories(env, num_of_trajs, horizon_cutoff, debug=debug)
        return states, actions, rewards


    def exp_2(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = self.auction_item_spec_pmf
        campaign_pmf = self.campaign_pmf
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1000
        num_items_per_timestep_max = 1001
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)

        agents = [
                    Agent("agent1", FixedBidPolicy(1.0, 1.0, auction_item_spec=auction_item_specs[0])), 
                    Agent("agent2", FixedBidPolicy(2.0, 2.0, auction_item_spec=auction_item_specs[1]))
                ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        # Vectorize is False
        env.vectorize = False
        env.init(agents, num_of_days)
        states, actions, rewards = generate_trajectories(env, num_of_trajs, horizon_cutoff, debug=debug)
        return states, actions, rewards


    def exp_3(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = self.auction_item_spec_pmf
        campaign_pmf = self.campaign_pmf
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 2
        num_items_per_timestep_max = 3
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)

        agents = [
                    Agent("agent1", BudgetPerReachPolicy()), 
                    Agent("agent2", BudgetPerReachPolicy())
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)
        states, actions, rewards = generate_trajectories(env, num_of_trajs, horizon_cutoff, debug=debug)
        return states, actions, rewards


    def exp_4(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 1,
                    self.auction_item_specs[1] : 1
        })
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 100
        num_items_per_timestep_max = 101
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
  
        agents = [
                    Agent("agent1", DummyREINFORCE(learning_rate=0.0002)), 
#                    Agent("agent2", FixedBidPolicy(0.5, 0.5, auction_item_spec=auction_item_specs[1]))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)
        
        # an array of shape [num_of_samples, num_of_agents, state_size]
        agent_states_samples = env.get_states_samples(10000)
        agent_states_samples = agent_states_samples.reshape(-1, *agent_states_samples.shape[2:])
        # scaler = ColumnTransformer([
        #                         ('0', MinMaxScaler(), [0]), 
        #                         ('1', MinMaxScaler(), [1]), 
        #                         #('2', 'passthrough', [2]), 
        #                         ('2', MinMaxScaler(), [2]), 
        #                         ('3', MinMaxScaler(), [3]), 
        #                         ('4', MinMaxScaler(), [4]), 
        #                         ('5', MinMaxScaler(), [5])
        # ])
        scaler = MinMaxScaler()
        scaler.fit(agent_states_samples)
        scale_states_func = lambda states: scaler.transform(states)
        # pprint(scaler.inverse_transform(scaler.transform([[10, 100, 1, 100, 0, 0]])))

        NUM_EPOCHS = 150
        agents[0].policy.build( (NUM_EPOCHS, num_of_trajs, len(agents) * len(agent_states_samples[0])) )
        agents[0].policy.summary()
        print("optimizer: {}, learning_rate: {}".format(agents[0].policy.optimizer, agents[0].policy.learning_rate))
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS))
        
        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff, scale_states_func=scale_states_func, debug=debug)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))
        
        end = time.time()

        avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
        print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        print("Took {} sec to train".format(end-start))

        return None, None, None


    def exp_5(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 1,
                    self.auction_item_specs[1] : 1
        })
        campaign_pmf = PMF({
                    self.campaigns[0] : 1,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 100
        num_items_per_timestep_max = 101
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
  
        agents = [
                    Agent("agent1", DummyREINFORCE(learning_rate=0.0002)), 
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)
        
        # an array of shape [num_of_samples, num_of_agents, state_size]
        agent_states_samples = env.get_states_samples(10000)
        agent_states_samples = agent_states_samples.reshape(-1, *agent_states_samples.shape[2:])
        scaler = MinMaxScaler()
        scaler.fit(agent_states_samples)
        scale_states_func = lambda states: scaler.transform(states)

        NUM_EPOCHS = 150
        agents[0].policy.build( (NUM_EPOCHS, num_of_trajs, len(agents) * len(agent_states_samples[0])) )
        agents[0].policy.summary()
        print("optimizer: {}, learning_rate: {}".format(agents[0].policy.optimizer, agents[0].policy.learning_rate))
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS))
        
        start = time.time()

        for i in range(NUM_EPOCHS):
            train(env, num_of_trajs, horizon_cutoff, scale_states_func=scale_states_func, debug=debug)
            avg_and_std_rwds = [(agent.name, np.mean(agent.cumulative_rewards[-num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-num_of_trajs:])) for agent in env.agents]
            print("epoch: {}, avg and std rwds: {}".format(i, avg_and_std_rwds))
        
        end = time.time()

        avg_and_std_rwds_last_50_epochs = [(agent.name, np.mean(agent.cumulative_rewards[-50*num_of_trajs:]), 
                            np.std(agent.cumulative_rewards[-50*num_of_trajs:])) for agent in env.agents]
        print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        print("Took {} sec to train".format(end-start))

        return None, None, None


    def exp_6(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_7(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_8(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_9(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_10(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Uniform_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_11(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Triangular_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_12(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_13(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Triangular_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_14(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Triangular_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None
    

    def exp_15(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Triangular_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None
    

    def exp_16(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Triangular_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_17(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Baseline_V_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr
                            ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None

      
    def exp_18(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Tabu_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None
                  
        
    def exp_19(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Tabu_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None

      
    def exp_20(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1",
                        AC_Q_Fourier_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_21(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1",
                        REINFORCE_Tabu_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_22(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1",
                        REINFORCE_Tabu_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_23(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1",
                        REINFORCE_AdaptiveLR_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_24(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1",
                        REINFORCE_Gaussian_v6_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_25(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1",
                        REINFORCE_Gaussian_v6_2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None
                      
        
    def exp_100(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]
        
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_101(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Baseline_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_102(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_TD_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_103(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_Q_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_104(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Uniform_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_105(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_106(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_SARSA_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]
        
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
         
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_107(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Baseline_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_108(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_TD_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]
        
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_109(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_Q_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_110(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_SARSA_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_111(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]
        
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_112(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Baseline_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_113(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_TD_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_114(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_Q_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_115(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Uniform_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_116(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_117(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_SARSA_Gaussian_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]
        
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
         
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_118(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", REINFORCE_Baseline_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_119(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_TD_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]
        
        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)
   
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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_120(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_Q_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None


    def exp_121(self, num_days, num_trajs, num_epochs, lr, debug=False):
        auction_item_specs = self.auction_item_specs
        auction = self.first_price_auction
        campaigns = self.campaigns
        auction_item_spec_pmf = PMF({
                    self.auction_item_specs[0] : 0,
                    self.auction_item_specs[1] : 1
        })
        auction_item_spec_ids = [self.auction_item_specs[1].uid]
        campaign_pmf = PMF({
                    self.campaigns[0] : 0,
                    self.campaigns[1] : 1
        })
        if debug:
            for c in campaigns:
                pprint(c)
                print()

        num_items_per_timestep_min = 1
        num_items_per_timestep_max = 2
        env = OneCampaignNDaysEnv(auction, auction_item_spec_pmf, campaign_pmf,
                                  num_items_per_timestep_min, num_items_per_timestep_max)
             
        agents = [
                    Agent("agent1", AC_SARSA_Triangular_MarketEnv_Continuous(auction_item_spec_ids, learning_rate=lr)),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        num_of_days = num_days # how long the game lasts
        num_of_trajs = num_trajs # how many times to run the game
        NUM_EPOCHS = num_epochs # how many batches of trajs to run
        horizon_cutoff = 100
        print("days per traj: {}, trajs per epoch: {}, EPOCHS: {}".format(num_of_days, num_of_trajs, NUM_EPOCHS)) 

        env.vectorize = True
        env.init(agents, num_of_days)

        scale_states_func, _  = self.get_states_scaler_descaler(env)
        scale_actions_func, descale_actions_func = self.get_actions_scaler_descaler(env)
        agents[0].set_scalers(scale_states_func, scale_actions_func)
        agents[0].set_descaler(descale_actions_func)

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
                print("Avg. of last 50 epochs: {}".format(avg_and_std_rwds_last_50_epochs))
        
        end = time.time()
        print("Took {} sec to train".format(end-start))
        return None, None, None

    
    def exp_400(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_2()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None

    
    def exp_1000(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1001(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1002(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1003(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1004(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None
   

    def exp_1005(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1006(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1007(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1008(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1009(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None

    def exp_1010(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1011(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1012(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1013(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1014(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1015(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1016(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1017(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1018(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1019(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1020(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Baseline_V_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1021(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Baseline_V_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1022(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Baseline_V_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1023(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Baseline_V_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1024(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Baseline_V_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1025(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_LogNormal_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_1026(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v5_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2000(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2001(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2002(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2003(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2004(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None
   

    def exp_2005(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2006(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2007(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2008(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2009(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None

    def exp_2010(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2011(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2012(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2013(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2014(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_2015(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(1, 1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None

    
    def exp_3000(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3001(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3002(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3003(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3004(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None
   

    def exp_3005(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_Q_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3006(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3007(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3008(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3009(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None

    def exp_3010(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3011(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3012(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3013(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Gaussian_v2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3014(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_3015(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v4_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=True
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", StepPolicy(num_days, -1))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4000(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_2()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4001(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_2()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4002(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_2()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4003(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_2()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_TD_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4004(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_2()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        AC_SARSA_Baseline_V_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4100(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_3()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4101(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_3()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Tabu_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4200(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_4()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4201(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_4()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Tabu_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4300(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_5()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4301(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_5()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Tabu_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4400(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_6()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_4401(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_6()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Tabu_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidPolicy(5, 5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5100(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_3()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5101(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_3()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Tabu_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5102(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_3()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v6_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5103(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_3()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v6_2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5104(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_3()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5105(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_3()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5200(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_4()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5201(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_4()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Tabu_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5202(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_4()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v6_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5204(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_4()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5300(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_5()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5301(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_5()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Tabu_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5302(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_5()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v6_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5304(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_5()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5400(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_6()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5401(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_6()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Baseline_Tabu_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5402(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_6()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v6_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_5404(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_6()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
                    Agent("agent2", FixedBidAdaptiveLimitPolicy(5))
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_9000(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v2_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_9001(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_9002(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v6_1_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_9003(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v6_2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_9004(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v6_3_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_9005(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_Gaussian_v3_2_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_9006(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_AdaptiveLR_v3_2_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
                        ),
                        scale_states_func, scale_actions_func,
                        descale_actions_func
                    ),
        ]

        # Run the game
        self.run(env, agents, num_days, num_trajs, num_epochs, 100, vectorize=True, debug=debug)
        return None, None, None


    def exp_9007(self, num_days, num_trajs, num_epochs, lr, debug=False):
        # Get a pre-defined environment
        env, auction_item_spec_ids = self.setup_1()
        
        # Get scaling/decaling info
        scale_states_func, actions_scaler, \
        scale_actions_func, descale_actions_func, scaled_avg_bpr = self.get_transformed(env)

        # Setup the agents of the game
        agents = [
                    Agent("agent1", 
                        REINFORCE_AdaptiveLR_v6_1_Gaussian_MarketEnv_Continuous(
                            auction_item_spec_ids, learning_rate=lr, 
                            budget_per_reach=scaled_avg_bpr, shape_reward=False
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
        'exp_1': experiment.exp_1,
        'exp_2': experiment.exp_2,
        'exp_3': experiment.exp_3,
        'exp_4': experiment.exp_4,
        'exp_5': experiment.exp_5,
        'exp_6': experiment.exp_6, # REINFORCE_Gaussian vs. FixedBidPolicy
        'exp_7': experiment.exp_7, # REINFORCE_Baseline_Gaussian vs. FixedBidPolicy
        'exp_8': experiment.exp_8, # AC_TD_Gaussian vs. FixedBidPolicy
        'exp_9': experiment.exp_9, # AC_Q_Gaussian vs. FixedBidPolicy
        'exp_10': experiment.exp_10, # REINFORCE_Uniform vs. FixedBidPolicy
        'exp_11': experiment.exp_11, # REINFORCE_Triangular vs. FixedBidPolicy
        'exp_12': experiment.exp_12, # AC_SARSA_Gaussian vs. FixedBidPolicy
        'exp_13': experiment.exp_13, # REINFORCE_Baseline_Triangular vs. FixedBidPolicy
        'exp_14': experiment.exp_14, # AC_TD_Triangular vs. FixedBidPolicy
        'exp_15': experiment.exp_15, # AC_Q_Triangular vs. FixedBidPolicy
        'exp_16': experiment.exp_16, # AC_SARSA_Triangular vs. FixedBidPolicy
        'exp_17': experiment.exp_17, # AC_SARSA_Baseline_V_Gaussian vs. FixedBidPolicy
        'exp_18': experiment.exp_18, # REINFORCE_Tabu_Gaussian (w/o rwd shaping) vs. FixedBidPolicy
        'exp_19': experiment.exp_19, # REINFORCE_Tabu_Gaussian (w rwd shaping) vs. FixedBidPolicy
        'exp_20': experiment.exp_20, # AC_Q_Fourier_Gaussian (w/o rwd shaping) vs. FixedBidPolicy
        'exp_21': experiment.exp_21, # REINFORCE_Tabu_Gaussian_v2 vs. FixedBidPolicy
        'exp_22': experiment.exp_22, # REINFORCE_Tabu_Gaussian_v3 vs. FixedBidPolicy
        'exp_23': experiment.exp_23, # REINFORCE_AdaptiveLR_Gaussian vs. FixedBidPolicy
        'exp_24': experiment.exp_24, # REINFORCE_Gaussian_v6_1 vs. FixedBidPolicy
        'exp_25': experiment.exp_25, # REINFORCE_Gaussian_v6_2 vs. FixedBidPolicy
        'exp_100': experiment.exp_100, # REINFORCE_Gaussian vs. StepPolicy (increasing)
        'exp_101': experiment.exp_101, # REINFORCE_Baseline_Gaussian vs. StepPolicy (increasing)
        'exp_102': experiment.exp_102, # AC_TD_Gaussian vs. StepPolicy (increasing)
        'exp_103': experiment.exp_103, # AC_Q_Gaussian vs. StepPolicy (increasing)
        'exp_104': experiment.exp_104, # REINFORCE_Uniform vs. StepPolicy (increasing)
        'exp_105': experiment.exp_105, # REINFORCE_Triangular vs. StepPolicy (increasing)
        'exp_106': experiment.exp_106, # AC_SARSA_Gaussian vs. StepPolicy (increasing)
        'exp_107': experiment.exp_107, # REINFORCE_Baseline_Triangular vs. StepPolicy (increasing)
        'exp_108': experiment.exp_108, # AC_TD_Triangular vs. StepPolicy (increasing)
        'exp_109': experiment.exp_109, # AC_Q_Triangular vs. StepPolicy (increasing)
        'exp_110': experiment.exp_110, # AC_SARSA_Triangular vs. StepPolicy (increasing)
        'exp_111': experiment.exp_111, # REINFORCE_Gaussian vs. StepPolicy (decreasing)
        'exp_112': experiment.exp_112, # REINFORCE_Baseline_Gaussian vs. StepPolicy (decreasing)
        'exp_113': experiment.exp_113, # AC_TD_Gaussian vs. StepPolicy (decreasing)
        'exp_114': experiment.exp_114, # AC_Q_Gaussian vs. StepPolicy (decreasing)
        'exp_115': experiment.exp_115, # REINFORCE_Uniform vs. StepPolicy (decreasing)
        'exp_116': experiment.exp_116, # REINFORCE_Triangular vs. StepPolicy (decreasing)
        'exp_117': experiment.exp_117, # AC_SARSA_Gaussian vs. StepPolicy (decreasing)
        'exp_118': experiment.exp_118, # REINFORCE_Baseline_Triangular vs. StepPolicy (decreasing)
        'exp_119': experiment.exp_119, # AC_TD_Triangular vs. StepPolicy (decreasing)
        'exp_120': experiment.exp_120, # AC_Q_Triangular vs. StepPolicy (decreasing)
        'exp_121': experiment.exp_121, # AC_SARSA_Triangular vs. StepPolicy (decreasing)
        'exp_400': experiment.exp_400, # env setup 2, REINFORCE_v2_Gaussian (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1000': experiment.exp_1000, # REINFORCE_v2_Gaussian (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1001': experiment.exp_1001, # REINFORCE_v2_Gaussian (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1002': experiment.exp_1002, # REINFORCE_v3_Gaussian (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1003': experiment.exp_1003, # REINFORCE_v3_Gaussian (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1004': experiment.exp_1004, # AC_Q_v2_Gaussian (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1005': experiment.exp_1005, # AC_Q_v2_Gaussian (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1006': experiment.exp_1006, # REINFORCE_Baseline_Gaussian_v2 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1007': experiment.exp_1007, # REINFORCE_Baseline_Gaussian_v2 (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1008': experiment.exp_1008, # REINFORCE_Baseline_Gaussian_v3 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1009': experiment.exp_1009, # REINFORCE_Baseline_Gaussian_v3 (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1010': experiment.exp_1010, # AC_TD_Gaussian_v2 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1011': experiment.exp_1011, # AC_TD_Gaussian_v2 (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1012': experiment.exp_1012, # AC_SARSA_Gaussian_v2 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1013': experiment.exp_1013, # AC_SARSA_Gaussian_v2 (w/ rwd shaping) vs. FixedBidPolicy
        'exp_1014': experiment.exp_1014, # REINFORCE_Gaussian_v4 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1015': experiment.exp_1015, # REINFORCE_Baseline_Gaussian_v4 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1016': experiment.exp_1016, # AC_TD_Gaussian_v3 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1017': experiment.exp_1017, # AC_TD_Gaussian_v4 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1018': experiment.exp_1018, # AC_Q_Gaussian_v3 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1019': experiment.exp_1019, # AC_Q_Gaussian_v4 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1020': experiment.exp_1020, # AC_SARSA_Baseline_V_Gaussian_v2 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1021': experiment.exp_1021, # AC_SARSA_Baseline_V_Gaussian_v3 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1022': experiment.exp_1022, # AC_SARSA_Baseline_V_Gaussian_v4 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1023': experiment.exp_1023, # AC_Q_Baseline_V_Gaussian_v2 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1024': experiment.exp_1024, # AC_Q_Baseline_V_Gaussian_v3 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1025': experiment.exp_1025, # REINFORCE_Baseline_LogNormal_v3_1 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_1026': experiment.exp_1026, # REINFORCE_Gaussian_v5 (w/o rwd shaping) vs. FixedBidPolicy
        'exp_2000': experiment.exp_2000, # REINFORCE_v2_Gaussian (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2001': experiment.exp_2001, # REINFORCE_v2_Gaussian (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2002': experiment.exp_2002, # REINFORCE_v3_Gaussian (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2003': experiment.exp_2003, # REINFORCE_v3_Gaussian (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2004': experiment.exp_2004, # AC_Q_v2_Gaussian (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2005': experiment.exp_2005, # AC_Q_v2_Gaussian (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2006': experiment.exp_2006, # REINFORCE_Baseline_Gaussian_v2 (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2007': experiment.exp_2007, # REINFORCE_Baseline_Gaussian_v2 (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2008': experiment.exp_2008, # REINFORCE_Baseline_Gaussian_v3 (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2009': experiment.exp_2009, # REINFORCE_Baseline_Gaussian_v3 (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2010': experiment.exp_2010, # AC_TD_Gaussian_v2 (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2011': experiment.exp_2011, # AC_TD_Gaussian_v2 (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2012': experiment.exp_2012, # AC_SARSA_Gaussian_v2 (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2013': experiment.exp_2013, # AC_SARSA_Gaussian_v2 (w/ rwd shaping) vs. StepPolicy (increasing)
        'exp_2014': experiment.exp_2014, # REINFORCE_Gaussian_v4 (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_2015': experiment.exp_2015, # REINFORCE_Baseline_Gaussian_v4 (w/o rwd shaping) vs. StepPolicy (increasing)
        'exp_3000': experiment.exp_3000, # REINFORCE_v2_Gaussian (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3001': experiment.exp_3001, # REINFORCE_v2_Gaussian (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3002': experiment.exp_3002, # REINFORCE_v3_Gaussian (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3003': experiment.exp_3003, # REINFORCE_v3_Gaussian (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3004': experiment.exp_3004, # AC_Q_v2_Gaussian (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3005': experiment.exp_3005, # AC_Q_v2_Gaussian (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3006': experiment.exp_3006, # REINFORCE_Baseline_Gaussian_v2 (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3007': experiment.exp_3007, # REINFORCE_Baseline_Gaussian_v2 (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3008': experiment.exp_3008, # REINFORCE_Baseline_Gaussian_v3 (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3009': experiment.exp_3009, # REINFORCE_Baseline_Gaussian_v3 (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3010': experiment.exp_3010, # AC_TD_Gaussian_v2 (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3011': experiment.exp_3011, # AC_TD_Gaussian_v2 (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3012': experiment.exp_3012, # AC_SARSA_Gaussian_v2 (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3013': experiment.exp_3013, # AC_SARSA_Gaussian_v2 (w/ rwd shaping) vs. StepPolicy (decreasing)
        'exp_3014': experiment.exp_3014, # REINFORCE_Gaussian_v4 (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_3015': experiment.exp_3015, # REINFORCE_Baseline_Gaussian_v4 (w/o rwd shaping) vs. StepPolicy (decreasing)
        'exp_4000': experiment.exp_4000, # REINFORCE_v2_1_Gaussian (w/o rwd shaping) vs. FixedBidPolicy, env setup 2
        'exp_4001': experiment.exp_4001, # REINFORCE_v3_1_Gaussian (w/o rwd shaping) vs. FixedBidPolicy, env setup 2
        'exp_4002': experiment.exp_4002, # REINFORCE_Baseline_Gaussian_v3_1 (w/o rwd shaping) vs. FixedBidPolicy, env setup 2
        'exp_4003': experiment.exp_4003, # AC_TD_Gaussian_v3_1 (w/o rwd shaping) vs. FixedBidPolicy, env setup 2
        'exp_4004': experiment.exp_4004, # AC_SARSA_Baseline_V_Gaussian_v3_1 (w/o rwd shaping) vs. FixedBidPolicy, env setup 2
        'exp_4100': experiment.exp_4100, # env setup 3, REINFORCE_Baseline_Gaussian_v3_1  vs. FixedBidPolicy
        'exp_4101': experiment.exp_4101, # env setup 3, REINFORCE_Baseline_Tabu_Gaussian_v3_1  vs. FixedBidPolicy
        'exp_4200': experiment.exp_4200, # env setup 4, REINFORCE_Baseline_Gaussian_v3_1  vs. FixedBidPolicy
        'exp_4201': experiment.exp_4201, # env setup 4, REINFORCE_Baseline_Tabu_Gaussian_v3_1 vs. FixedBidPolicy
        'exp_4300': experiment.exp_4300, # env setup 5, REINFORCE_Baseline_Gaussian_v3_1  vs. FixedBidPolicy
        'exp_4301': experiment.exp_4301, # env setup 5, REINFORCE_Baseline_Tabu_Gaussian_v3_1 vs. FixedBidPolicy
        'exp_4400': experiment.exp_4400, # env setup 6, REINFORCE_Baseline_Gaussian_v3_1  vs. FixedBidPolicy
        'exp_4401': experiment.exp_4401, # env setup 6, REINFORCE_Baseline_Tabu_Gaussian_v3_1 vs. FixedBidPolicy
        'exp_5100': experiment.exp_5100, # env setup 3, REINFORCE_Baseline_Gaussian_v3_1  vs. FixedBidAdaptiveLimitPolicy
        'exp_5101': experiment.exp_5101, # env setup 3, REINFORCE_Baseline_Tabu_Gaussian_v3_1  vs. FixedBidAdaptiveLimitPolicy
        'exp_5102': experiment.exp_5102, # env setup 3, REINFORCE_Gaussian_v6_1 vs. FixedBidAdaptiveLimitPolicy
        'exp_5103': experiment.exp_5103, # env setup 3, REINFORCE_Gaussian_v6_2 vs. FixedBidAdaptiveLimitPolicy
        'exp_5104': experiment.exp_5104, # env setup 3, REINFORCE_v2_1_Gaussian vs. FixedBidAdaptiveLimitPolicy
        'exp_5105': experiment.exp_5105, # env setup 3, REINFORCE_v3_1_Gaussian vs. FixedBidAdaptiveLimitPolicy
        'exp_5200': experiment.exp_5200, # env setup 4, REINFORCE_Baseline_Gaussian_v3_1  vs. FixedBidAdaptiveLimitPolicy
        'exp_5201': experiment.exp_5201, # env setup 4, REINFORCE_Baseline_Tabu_Gaussian_v3_1 vs. FixedBidAdaptiveLimitPolicy
        'exp_5202': experiment.exp_5202, # env setup 4, REINFORCE_Gaussian_v6_1 vs. FixedBidAdaptiveLimitPolicy
        'exp_5204': experiment.exp_5204, # env setup 4, REINFORCE_v2_1_Gaussian vs. FixedBidAdaptiveLimitPolicy
        'exp_5300': experiment.exp_5300, # env setup 5, REINFORCE_Baseline_Gaussian_v3_1  vs. FixedBidAdaptiveLimitPolicy
        'exp_5301': experiment.exp_5301, # env setup 5, REINFORCE_Baseline_Tabu_Gaussian_v3_1 vs. FixedBidAdaptiveLimitPolicy
        'exp_5302': experiment.exp_5302, # env setup 5, REINFORCE_Gaussian_v6_1 vs. FixedBidAdaptiveLimitPolicy
        'exp_5304': experiment.exp_5304, # env setup 5, REINFORCE_v2_1_Gaussian vs. FixedBidAdaptiveLimitPolicy
        'exp_5400': experiment.exp_5400, # env setup 6, REINFORCE_Baseline_Gaussian_v3_1  vs. FixedBidAdaptiveLimitPolicy
        'exp_5401': experiment.exp_5401, # env setup 6, REINFORCE_Baseline_Tabu_Gaussian_v3_1 vs. FixedBidAdaptiveLimitPolicy
        'exp_5402': experiment.exp_5402, # env setup 6, REINFORCE_Gaussian_v6_1 vs. FixedBidAdaptiveLimitPolicy
        'exp_5404': experiment.exp_5404, # env setup 6, REINFORCE_v2_1_Gaussian vs. FixedBidAdaptiveLimitPolicy
        'exp_9000': experiment.exp_9000, # env setup 1, REINFORCE_v2_1_Gaussian vs. None
        'exp_9001': experiment.exp_9001, # env setup 1, REINFORCE_v3_1_Gaussian vs. None
        'exp_9002': experiment.exp_9002, # env setup 1, REINFORCE_v6_1_Gaussian vs. None
        'exp_9003': experiment.exp_9003, # env setup 1, REINFORCE_v6_2_Gaussian vs. None
        'exp_9004': experiment.exp_9004, # env setup 1, REINFORCE_v6_3_Gaussian vs. None
        'exp_9005': experiment.exp_9005, # env setup 1, REINFORCE_v3_2_Gaussian vs. None
        'exp_9006': experiment.exp_9006, # env setup 1, REINFORCE_AdaptiveLR_v3_2_Gaussian vs. None
        'exp_9007': experiment.exp_9007, # env setup 1, REINFORCE_AdaptiveLR_v6_1_Gaussian vs. None
    }
    try:
        exp_func = function_mappings[exp]
    except KeyError:
        raise ValueError('invalid input')

    print("Running experiment {}".format(exp_func.__name__))
    states, actions, rewards = exp_func(num_days, num_trajs, num_epochs, lr, debug=debug)
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