
from os import link
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class TrafficMidEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TrafficMidEnv, self).__init__()

        #network with 2-2 links setting and one OD pair

        #  (link 0)   (link 2)
        # O---<=>---A---<=>---D
        #  (link 1)   (link 3)

        self.action_space = spaces.Box(low=np.array([10.0, 1.0, 1.0, 1.0]), high=np.array([+10.0, +1.0, +1.0, +1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0, 0, 0, 0]), high=np.array([2000, 2000, 2000, 2000]), dtype=np.float32)

        self.num_link = 4
        
        # self.total_veh_num = 600
        self.state = np.array([500, 100, 500, 100])         #veh_num
        self.lanes_link = np.array([2, 2, 2, 2])
        self.length_link = np.array([500, 500, 500, 500])  #meters
        self.free_v_link = np.array([30, 30, 30, 30])      #meters per second
        self.alpha_link = np.array([0.4, 0.4, 0.4, 0.4])
        self.jam_density_link = np.array([2.0, 2.0, 2.0, 2.0])    # num_of_veh per meter
        self.human_headway_link = np.array([3.0, 3.0, 3.0, 3.0])



        #dynamic coef, which needed to be twitched from 1
        self.miu = 0.05
        self.nu = 1000  #new coefficient for hybrid reward function

    def step(self, action):
        flag_done = False
        # print("Action:", action)
        # print("State:",self.state)


        veh_num = self.state.copy()
        density_link = veh_num/self.length_link
        
        cri_density_link = self.lanes_link/(action*self.alpha_link + self.human_headway_link*(1-self.alpha_link))
        flow_link_in = np.zeros(self.num_link)
        flow_link_out = np.zeros(self.num_link)
        flow_on_link = np.zeros(self.num_link)
        latency_link = np.zeros(self.num_link)


        reward_cumulation = 0
        demand = np.zeros(2)
        demand[0] = 22
        # demand = self.free_v_link*cri_density_link


        for train_loop in range(100):

            #calculate input flow at origin
            for i in range(2):
                flow_link_in[i] = max(demand[i],0)


            #calculate output flow at midpoint
            flow_out_midpoint = 0
            for i in range(2):
                if(density_link[i]<cri_density_link[i]):
                    flow_link_out[i] = self.free_v_link[i]*density_link[i]

                elif(density_link[i]>self.jam_density_link[i]):
                    flow_link_out[i] = 0

                else:
                    flow_link_out[i] = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
                    
                flow_link_out[i] = min(veh_num[i], flow_link_out[i])

                #update veh_num and density on link_0 and link_1
                veh_num[i] = veh_num[i] + flow_link_in[i] - flow_link_out[i]
                density_link[i] = veh_num[i]/self.length_link[i]

                flow_out_midpoint += flow_link_out[i]
            

            #divide flow_out_midpoint onto link_2 and link_3
            link_2_num = veh_num[2]
            link_3_num = veh_num[3]

            link_2_val = link_2_num*np.exp(-self.miu*(latency_link[2]))
            link_3_val = link_3_num*np.exp(-self.miu*(latency_link[3]))

            total_val = link_2_val+link_3_val

            flow_link_in[2] = flow_out_midpoint*link_2_val/(total_val)
            flow_link_in[3] = flow_out_midpoint*link_3_val/(total_val)
            



            #calculate output flow at destination
            flow_out_dest = 0
            for j in range(2):
                i = j+2
                if(density_link[i]<cri_density_link[i]):
                    flow_link_out[i] = self.free_v_link[i]*density_link[i]

                elif(density_link[i]>self.jam_density_link[i]):
                    flow_link_out[i] = 0

                else:
                    flow_link_out[i] = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
                    
                flow_link_out[i] = min(veh_num[i], flow_link_out[i])

                #update veh_num and density on link_2 and link_3
                veh_num[i] = veh_num[i] + flow_link_in[i] - flow_link_out[i]
                density_link[i] = veh_num[i]/self.length_link[i]

                flow_out_dest += flow_link_out[i]

            


            #update flow on link and corresponding latency
            for i in range(self.num_link):
                if(density_link[i]<cri_density_link[i]):
                    flow_on_link[i] = self.free_v_link[i]*density_link[i]
                    latency_link[i] = self.length_link[i]/self.free_v_link[i]

                elif(density_link[i]>self.jam_density_link[i]):
                    flow_on_link[i] = 0
                    # latency_link[i] = np.infty
                    latency_link[i] = 10000000000
                    # flag_done = True # failed
                    # print("Link ",str(i), "is totally jammed!")

                else:
                    flow_on_link[i] = self.free_v_link[i]*cri_density_link[i]*(self.jam_density_link[i]-density_link[i])/(self.jam_density_link[i]-cri_density_link[i])
                    latency_link[i] = self.length_link[i]*(self.jam_density_link[i]/flow_on_link[i]+(cri_density_link[i]-self.jam_density_link[i])/(self.free_v_link[i]*cri_density_link[i]))               
        

            reward_cumulation -= sum(density_link)


            #TODO: Update demand for next timestep at origin by choice dynamics

            path_1_num = veh_num[0]
            path_2_num = veh_num[1]


            latency_next_node = min(latency_link[2], latency_link[3])
            path_1_val = path_1_num*np.exp(-self.miu*(latency_link[0]+latency_next_node))
            path_2_val = path_2_num*np.exp(-self.miu*(latency_link[1]+latency_next_node))

            total_val = path_1_val+path_2_val

            _path_1 = sum(demand)*path_1_val/(total_val)
            _path_2 = sum(demand)*path_2_val/(total_val)

            demand[0] = _path_1
            demand[1] = _path_2


            # if(train_loop<10):
            print("Train loop:", train_loop)
            print("Action:", action)
            # print("total divide:", total_val)
            print("Flow_in:", flow_link_in)
            print("Flow_out:", flow_link_out)
            print("Flow on link:", flow_on_link)
            print("Latencies:", latency_link)
            print("veh_num:",veh_num)
            print("New demand:", demand)
            print("Density:", density_link)
            # print("Vals:", path_1_val, path_2_val)
            # print("Path_Val:", np.array([path_1_val, path_2_val, path_3_val, path_4_val]))
            print("\n")

       

        reward = reward_cumulation/100

        print("Veh num after 100 timesteps:",veh_num) 

        info = {}
        done = False

        return self.state, reward, done, info

    def reset(self):
        # reset state to starting case

        # for 2-2 links setting with one OD pair
        self.state = np.array([500, 100, 500, 100])
        return self.state
  
    def render(self, mode='human'):
        pass

    def close(self):
        pass