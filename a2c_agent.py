import numpy as np
import gym
import a2c_net 

#Hybrid network for actor-critic
#Batch a2c with n-steps rewards

class create_agent(object):
    
    def __init__(self,batch_dim,env,network,nsteps,gamma = 0.90):
        self.batch_dim = batch_dim
        self.env = env
        self.network = network
        self.discount = gamma
        self.sum_eps_rwd = []
        self.count_episodes = 0
        self.n_steps = nsteps 

    def reset(self):
        self.sum_eps_rwd = []
        self.count_episodes = 0

    def get_average_rewards(self, batch_dim):
        average_rew = np.mean(self.sum_eps_rwd[-100:])
        print("Mean Rewards: ", average_rew, "Episode: ", self.count_episodes, "Num of episodes in batch: ", batch_dim)
        return average_rew

    
    def set_nsteps(self,new_nstep):
        #print("new_nstep: ", new_nstep)
        self.n_steps = new_nstep

    # Collect a batch experience usefull to calculate QTarget and Advantage
    # We precalculate cumulative rewards and cumulative next state target to accellerate the batch_process via vectorialization
    #####
    # Return will be:
    #   batch: a list of 4 elements
    #       batch_states[]: shape(1,self.batch_dim)
    #       batch_actions[]: shape(1,self.batch_dim)
    #       cumulative_rewards[] = shape(1,self.batch_dim) -> precalculated cumulative discounted rewards
    #       cumulative_next_states[] = shape(1,self.batch_dim) -> a list of state far n-step from each "current state" in batch_states vector
    #       discount_cum_next_states[] = shape(1,self.batch_dim) -> list of gammas raised to n (where the value of n change contextually to the step episode). 
    #

    def sample_batch(self):
        batch_steps = 0
        cumulative_rewards =  []
        batch_states = []
        batch_actions = []
        cumulative_next_states = []
        discount_cum_next_states = []
        step_to_cumulate = 0
        while batch_steps <= self.batch_dim:         
            
            # Collect all episode rewards. It's usefull to calculate cumulative rewards and it is resetted at each episode
            episode_rew  = [] 

            # Counter of steps that have received their cumulative reward
            cumulative_index = 0
            
            ob = self.env.reset()
            episode_step = 0
            total_ep_rew = 0
            done = False

            # Continue until we finish to filling the cumulative vectors. 
            # In the initial case cumulative_index is equal to episode_step but only because we are in the initial step
            while cumulative_index <= episode_step or episode_step == 0:

                # If the episode is not finished keep step forward
                if not done and episode_step < self.env.spec.max_episode_steps:
                    batch_states.append(ob)
                    act = self.network.get_action(ob.reshape(1, -1))
                    ob,rw,done,_ = self.env.step(act)
                    total_ep_rew += rw
                    batch_actions.append(act)
                    episode_rew.append(rw)

                # Count the number of steps I have to go back to redistribute the reward without overwriting observations that have already been calculated 
                step_to_cumulate = min(self.n_steps,episode_step-cumulative_index) 
                
                # When excecute enough step start to cumulate rewards
                if episode_step >= self.n_steps:
                    
                    # vectorize the cumulative reward loop
                    cumulative_gamma = np.full(step_to_cumulate+1, self.discount)
                    
                    # The exponents of gamma are go from 0 to step_to_cumulate
                    exp_gamma = np.arange(step_to_cumulate+1)
                    cumulative_gamma = cumulative_gamma ** exp_gamma

                    # Calculate the Cumulative Discounted Reward 
                    assert cumulative_index+step_to_cumulate <= episode_step, "cumulative_index+step_to_cumulate IS NOT <= episode_step"
                    cumulative_rewards.append(np.dot(episode_rew[cumulative_index:cumulative_index+step_to_cumulate+1], cumulative_gamma))
                    
                    # Calculate the Cumulative n-step_next State
                    cumulative_next_states.append(batch_states[episode_step])

                    # We construct the vector of gammas raised to the n (we multiply these vectors after we estimates the next_states values)
                    if step_to_cumulate > 0:
                        discount_cum_next_states.append(self.discount ** step_to_cumulate)
                    else:
                    # If this is the very last step there is no next_state. So we eliminate this value by multiply it with a gamma = 0
                        discount_cum_next_states.append(0)
                    
                    cumulative_index += 1 

                if not done and episode_step < self.env.spec.max_episode_steps:
                    episode_step += 1

            #end of episode
            self.env.close()
            batch_steps += episode_step
            self.count_episodes+= 1
            self.sum_eps_rwd.append(total_ep_rew)

        batch = [cumulative_rewards,batch_states,batch_actions,cumulative_next_states,discount_cum_next_states]
        assert len(batch_states) == len(batch_actions) == len(cumulative_rewards) == len(cumulative_next_states) == len(discount_cum_next_states), "Problem with batch size in sample batch "
        return batch
    

    # Optimization: we want to stabilize the updating of critic network (similar case of the fixed q-target solution) 
    # we take a few (num_grad_steps_per_target_update) gradient step in direction of the qtarget parameter
    # every num_grad_steps_per_target_update we recompute a new qtarget with the updated critic network 
    # e recompute a few target step in that direction.
    def update_critics(self,batch,qtarget):
        """
            batch: the batch of data collect with the current policy. We need this to update the qtarget
            qtarget: the qtarget calculated for all episode in batch. shape(1, batch_dim) -> da verificare 
        """
        
        # These value shuld be hyperparameters
        num_grad_steps_per_target_update = 10
        num_target_updates = 10

        # Join togheter the obs from all episodes in the batch
        _,batch_obs,_,_,_0 = batch
        
        for i in range(num_target_updates * num_grad_steps_per_target_update):
        
            # Regress onto targets to update value function by taking a few gradient steps
            self.network.update_critic(batch_obs,qtarget)
        
            # Every num_grad_steps_per_target_update steps, recompute the target values
            if i % num_grad_steps_per_target_update == 0: 
        
                # Update targets with current value function    
                qtarget,_ = self.process_batch(batch,adv_request=False)
        
    def update_actor(self,batch,adv):
        _,_,batch_acts,_,_ = batch
        _,batch_obs,_,_,_0 = batch
        self.network.update_actor(batch_obs,batch_acts,adv)
    
    # This function calculate Q-Target and Advantage for the entire batch
    # batch: a collection of data in the batch 
    # adv: in case you want only the Q-target set this parameter to False
    def process_batch(self,batch, adv_request = True):
        cumulative_rewards,batch_states,_,cumulative_next_states,discount_cum_next_states = batch

        # Initialize Adv and QTarget
        Adv = []
        QTarget = []

        ##### Batch Q-Target #####            
        discounted_next_state = self.network.get_state_value(cumulative_next_states) * discount_cum_next_states
        QTarget = cumulative_rewards + discounted_next_state

        ##### Batch ADV ##### 
        # The function is the same for Q-Target and Advantage but in some case you just need a new Q-Target (see Update Critics)
        if adv_request: 
            Adv = QTarget - self.network.get_state_value(batch_states)
            
        return QTarget,Adv


   
            
        
    def train(self, max_iterations):
        open_ai_solve_condition = False
        itr = 0
        while itr <= max_iterations and not open_ai_solve_condition:   
            #print("init train")         
            # Collect a Batch of data = [obs,acts,rewards,next_obs]
            batch = self.sample_batch() 
            #print("collected batch")

            # Calculates Qtarget (to update critic) and Adv (to update actor)
            QTarget,Adv = self.process_batch(batch) 
            #print("processed batch")

            self.update_critics(batch,QTarget)
            #print("update critic")

            self.update_actor(batch,Adv)
            #print("updata actor")
            
            # Test if the env is already solved
            open_ai_condition = self.get_average_rewards(len(batch)) >= open_ai_solve_condition

            itr += 1
            print("Iteration: ", itr)