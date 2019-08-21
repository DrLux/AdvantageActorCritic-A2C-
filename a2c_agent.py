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

    def sample_batch(self):
        batch_steps = 0
        episodes, obs, acts, rewards, next_obs = [], [], [], [], []
        while batch_steps <= self.batch_dim:         
            ob = self.env.reset()
            episode_step = 0
            total_ep_rew = 0
            done = False
            while not done and episode_step < self.env.spec.max_episode_steps: # --> mettere min(self.env.spec.max_episode_steps e un parametro max_steps_per_episode)
                obs.append(ob)
                act = self.network.get_action(ob.reshape(1, -1))
                ob,rw,done,_ = self.env.step(act)
                total_ep_rew += rw
                acts.append(act)
                rewards.append(rw)
                next_obs.append(ob)
                episode_step += 1
            
            episodes.append([obs,acts,rewards,next_obs])
            self.env.close()
            batch_steps += episode_step
            self.count_episodes+= 1
            self.sum_eps_rwd.append(total_ep_rew)
        return episodes
    

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
        batch_obs = np.concatenate([episodes[0] for episodes in batch])
        
        for i in range(num_target_updates * num_grad_steps_per_target_update):
        
            # Regress onto targets to update value function by taking a few gradient steps
            self.network.update_critic(batch_obs,qtarget)
        
            # Every num_grad_steps_per_target_update steps, recompute the target values
            if i % num_grad_steps_per_target_update == 0: 
        
                # Update targets with current value function    
                qtarget,_ = self.process_batch(batch,adv=False)
        
    def update_actor(self,batch,adv):
        batch_acts = np.concatenate([episodes[1] for episodes in batch])
        batch_obs = np.concatenate([episodes[0] for episodes in batch])
        self.network.update_actor(batch_obs,batch_acts,adv)
    
    # This function calculate Q-Target and Advantage for the entire batch
    # episodes: a collection of data in the batch (in a form of list of episodes. Each episode contain a list of obs,actions,rewards and the new_obs)
    # adv: in case you want only the Q-target set this parameter to False
    def process_batch(self,episodes, adv = True):
        #print("init process batch")
        QTarget = []
        Adv = []
        batch_dim = 0
        batch_Adv = []

        # for each episode
        for e in range(len(episodes)): 

            # Extract the current episode's data
            obs, acts, rewards, next_obs = episodes[e]
 
            len_ep = len(obs)
            batch_dim += len_ep

            # N-step can't be bigger then the entire episode
            max_n_step = min(self.n_steps, len_ep)

        ##### The entire Q-Target episode #####
            
            # Calculate the cumulative reward for each episode step and collect result in this vector 
            # len(target_rews) = len_ep
            # formula: summation from t to t' of gamma*r(s,a)
            target_rews = self.calculate_target_rew_vect(len_ep,rewards)

            # Calculate the new_state's value
            # Collect the estimates of next n-step_next_values until the (end of episode - n-steps). 
            # From there, the other next step remain the last one repeated (can't go other last step)
            # In the very last step the value must be 0 
            # 
            # Formula: V(s_t+n)
            #
            # shape = (len_episode, dim_obs)
            new_st_value = next_obs[max_n_step:len_ep] + [next_obs[-1]]*max_n_step
            # len(new_st_value) = len_ep
            new_st_value = self.network.get_state_value(new_st_value)
            
            # Vector of gamma raised to n-step 
            # From len_ep - n-step, the exponent start to decrease until reach 0 in the very last step 
            #  len(gamma) = gamma
            exp_gamma = list(np.full(len_ep-max_n_step,max_n_step)) + list(range(max_n_step,0,-1))
            gamma = np.power(np.full(len_ep,self.discount),exp_gamma)
            
            # Mutliply the new_state_value to gamma  
            new_st_value = new_st_value * gamma

            #Calculate QTarget
            episode_qtarget = target_rews + new_st_value
            
            QTarget.append(episode_qtarget)

            ##### ADV #####
            
            # The function is the same for Q-Target and Advantage but in some case you just need a new Q-Target (see Update Critics)
            if adv: 
                # Calculate the current state value
                # shape = scalar
                current_s_value = self.network.get_state_value(obs)#.reshape(1, -1)
                Adv.append(episode_qtarget - current_s_value)
                batch_Adv = np.concatenate([qt for qt in Adv])
                assert(batch_dim == len(batch_Adv), "len(Batch_Adv) do not correnpond to Batch_size")
                #print("for ",t," current_s_value: ", current_s_value, " and QTarget[t]: ", QTarget[t], " Adv[t]: ", Adv[t], "\n")
        
        batch_QTarget = np.concatenate([qt for qt in QTarget])
        assert(batch_dim == len(batch_QTarget), "len(Batch_QTarget) do not correnpond to Batch_size")

        return batch_QTarget,batch_Adv


    ##### Q-Target ####
    # This function calculate the Q-target using the n-step trick to cumulate rewards
    # Parameters are:
    # len_ep: the episode's length
    # rewards: a list of all rewards for the current episode in the batch. shape(1,?len episode?)
    ###
    # return: the collection of summed rewards. shape()
    ###
 
    def calculate_target_rew_vect(self,len_ep,rewards): 
        # collect all the forwarded target reward in episode 
        # len(episode_target_rew) = len_ep
        episode_target_rew = [] 
        
        # for each episode's step            
        for t in range(len_ep): 
            # the last step of the summation, that is the minimum between n_steps and length of the remaining episode's steps
            step_to_go = min(self.n_steps, (len_ep-1)-t)
            
        # Vectorize loop for rewards operations for gamma^(t-t)r(s, a)
            # Create vector of gamma
            # shape: (step_to_go,)
            discount_vector = np.full(step_to_go, self.discount)
            
            # Creare a vector of gamma's exponents
            # shape: (step_to_go,)
            exp_discount_for_rew = np.arange(step_to_go)
            
            # Apply the exponents on gamma's vector
            # shape: (step_to_go,)
            discount_vector = np.power(discount_vector, exp_discount_for_rew)
                
        # Extract the slice of rewards from t to step_to_go
            rews = rewards[t:t+step_to_go]

        # Calculate the sum of rewards (end of n-step loop)
            target_rew = np.dot(rews,discount_vector)
            episode_target_rew.append(target_rew)

        return episode_target_rew        
            
        
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
            #print("Iteration: ", itr)