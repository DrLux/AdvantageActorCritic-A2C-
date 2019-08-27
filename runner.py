import numpy as np
import gym
import a2c_agent 
import a2c_net 
from gym.wrappers import Monitor
import matplotlib.pyplot as plt

### To DO
# - aggiungere un limite di steps
# - aggiungere la funzione di creazioen dell' env che mi fa scegliere se renderizzare o registrare
# - aggiungere il supporto a tensorboard 
# - migliorare il reset del grafo di tensorflow
# - aggiungere l'opzione di checkpoint per tensorflow
# - aggiungere Gae e td(lambda)
# - avere le simulazioni sincrone è un suicidio, devo eseguirle in parallelo
###


def plot_results(rewards_per_episode,different_nsteps,batch_dim):
        plt.figure(figsize=(12,8))
        for i in range(rewards_per_episode.shape[0]):
                plt.plot(rewards_per_episode[i], label='n-steps=' + str(different_nsteps[i]))

        plt.title('Average Rewards for n-Step A2C with batch size'+str(batch_dim))
        plt.legend(loc='best')
        plt.ylabel('Rewards')
        plt.show()
        plt.savefig('n_steps.png')

        
def create_env(name_env, wrapped):
    env = gym.make(name_env)
    if wrapped:
        env = Monitor(env, './video', force=True)
        return env

def main():
        ### Set Hyperparameters
        batch_dim = 1000
        iterations = 100
        discount_factor = 0.999
        name_env = 'CartPole-v0'

        # Indicate the treshold to reach in order to consider the task solved (https://github.com/openai/gym/wiki/Leaderboard)
        open_ai_baseline = 195 

        # Try a renge of different N-Steps = [100, 20, 1, 10, 50]
        different_nsteps = [10,100]
        
        n_sim = 1

        ## Initialize Variables
        rewards_per_episode = np.zeros((len(different_nsteps), iterations))
        env = gym.make(name_env)

        ## Extract info from env
        discrete = isinstance(env.action_space, gym.spaces.Discrete) 
        ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]

        ## Create objects of Network and Agent
        network = a2c_net.build_network(obs_dim, ac_dim,discrete) 
        agent = a2c_agent.create_agent(batch_dim,env,network,1,discount_factor)


        # Getting results for benchmark
        for ns in range(0,len(different_nsteps)):
                print("Start with: ", different_nsteps[ns])
                agent.set_nsteps(different_nsteps[ns])
                for i in range(0,n_sim):
                        print("Simulation number: ", i)
                        
                        # reset objects
                        network.init_tf_sess()
                        agent.reset()
                        
                        # *Core function* 
                        agent.train(iterations, open_ai_baseline)
                        rewards_per_episode[ns] += agent.get_average_rewards()
        env.close()
        rewards_per_episode /= n_sim
        plot_results(rewards_per_episode,different_nsteps,batch_dim)

if __name__ == "__main__":
    main()