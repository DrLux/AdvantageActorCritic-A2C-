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
        obs, acts, rewards, next_obs, n_terminal = [], [], [], [], []
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
                n_terminal.append(0)
                
            #if we are escaped from loop means that the last state for a terminal one so we update our n_terminal vector
            n_terminal[-1] = 1 
            self.env.close()
            batch_steps += episode_step
            self.count_episodes+= 1
            self.sum_eps_rwd.append(total_ep_rew)

        obs = np.array(obs, dtype=np.float32)
        acts = np.array(acts, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_obs = np.array(next_obs, dtype=np.float32)
        n_terminal = np.array(n_terminal, dtype=np.float32)

        batch = [obs, acts, rewards, next_obs, n_terminal]
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
        batch_obs,_,_,_,_ = batch
        
        for i in range(num_target_updates * num_grad_steps_per_target_update):
        
            # Regress onto targets to update value function by taking a few gradient steps
            self.network.update_critic(batch_obs,qtarget)
        
            # Every num_grad_steps_per_target_update steps, recompute the target values
            if i % num_grad_steps_per_target_update == 0: 
        
                # Update targets with current value function    
                qtarget,_ = self.process_batch(batch,adv=False)
        
    def update_actor(self,batch,adv):
        batch_obs, batch_acts, _, _, _ = batch
        self.network.update_actor(batch_obs,batch_acts,adv)
    
    # This function calculate Q-Target and Advantage for the entire batch
    # episodes: a collection of data in the batch (in a form of list of episodes. Each episode contain a list of obs,actions,rewards and the new_obs)
    # adv: in case you want only the Q-target set this parameter to False
    def process_batch(self,batch, adv = True):
        
        obs, acts, rewards, next_obs, n_terminal = batch

        qvalue = rewards + (1-n_terminal)*self.discount*self.network.get_state_value(next_obs)
        estimate_adv = None
                
        if adv: 
            estimate_adv = qvalue - self.network.get_state_value(obs)

        return qvalue,estimate_adv
 
        
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