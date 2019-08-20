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
        QTarget = []
        Adv = []

        # for each episode
        for e in range(len(episodes)): 

            # Extract the current episode's data
            obs, acts, rewards, next_obs = episodes[e]

            len_ep = len(obs)

            # for each episode's step            
            for t_p in range(len_ep): 

                # the last step of the summation, that is the minimum between n_steps and length of the remaining episode's steps
                # sh: 1
                last_step = min(self.n_steps, (len_ep-1)-t_p)
                
                ##### Q-Target #####
                qtarget = self.calculate_qtarget(t_p,last_step,rewards,next_obs)  
                QTarget.append(qtarget)

                ##### ADV #####
                
                # The function is the same for Q-Target and Advantage but in some case you just need a new Q-Target (see Update Critics)
                if adv: 
                    # Calculate the current state value
                    # shape = scalar
                    current_s_value = self.network.get_state_value(obs[t_p].reshape(1, -1))
                    Adv.append(qtarget - current_s_value)
                    #print("for ",t," current_s_value: ", current_s_value, " and QTarget[t]: ", QTarget[t], " Adv[t]: ", Adv[t], "\n")
                                
        return QTarget,Adv


    ##### Q-Target ####
    # This function calculate the Q-target using the n-step trick to cumulate rewards
    # Parameters are:
    # t_p: the actual step in the episode
    # last_step: the step that we want reaching in cumulating reward. We make sure che this value is <= ?len episode?
    # rewards: a list of all rewards for the current episode in the batch. shape(1,?len episode?)
    # next_obs: a list of obs for the current episode collected in the batch. shape(1,?len episode?) 

    def calculate_qtarget(self,t_p,last_step,rewards,next_obs):            
    
    # Vectorize loop for rewards operations for gamma^(t_p-t)r(s, a)
        # Create vector of gamma
        # shape: (1, last_step)
        discount_vector = np.full(last_step, self.discount)
        
        # Creare a vector of gamma's exponents
        # shape: (1, last_step)
        exp_discount_for_rew = np.arange(last_step)
        
        # Apply the exponents on gamma's vector
        discount_vector = np.power(discount_vector, exp_discount_for_rew)
        
    # Extract the slice of rewards from t to last_step
        rews = rewards[t_p:t_p+last_step]

    # Calculate the sum of rewards
        target_rew = np.sum(rews * discount_vector)

        # Calculate the new_state's value
        # shape = scalar
        new_st_value = self.network.get_state_value(next_obs[t_p+last_step].reshape(1, -1))

        # Mutliply the new_state_value to gamma  
        new_st_value *= self.discount ** last_step

    #Calculate QTarget
        qtarget = target_rew + new_st_value

        return qtarget        
            
        
    def train(self, num_episodes):
        open_ai_solve_condition = False
        while self.count_episodes <= num_episodes and not open_ai_solve_condition:            
            # Collect a Batch of data = [obs,acts,rewards,next_obs]
            batch = self.sample_batch() 
            
            # Calculates Qtarget (to update critic) and Adv (to update actor)
            QTarget,Adv = self.process_batch(batch) 
            
            self.update_critics(batch,QTarget)
            self.update_actor(batch,Adv)
            
            # Test if the env is already solved
            open_ai_condition = self.get_average_rewards(len(batch)) >= open_ai_solve_condition