class Env(object):
    """
    Original state:
    (*:Agent,#:devil,$:money)
    easy:
    *^^^
    ^^#^
    ^#$^
    ^^^^
    hard:
    *^#$
    ^^#^
    ^^^^
    ^^^^
    """
    def __init__(self,layout):
        self.size=4
        self.agent_position=[0,0]
        if layout=='easy':
            self.bad1_position=[1,2]
            self.bad2_position=[2,1]
            self.good_position=[2,2]

        elif layout=='hard':
            self.bad1_position=[0,2]
            self.bad2_position=[1,2]
            self.good_position=[0,3]

        self.action_space=4
        self.observation_space=2


    def render(self):
        for i in range(self.size):
            for j in range(self.size):
                if [i,j]==self.agent_position:
                    print('*',end=' ')
                elif [i,j]==self.bad1_position or [i,j]==self.bad2_position:
                    print('#',end=' ')
                elif [i,j]==self.good_position:
                    print('$',end=' ')
                else:
                    print('^',end=' ')
            print(' ')
        print(' ')


    def reset(self):
        self.agent_position=[0,0]
        return self.agent_position


    def step(self,action):
        '''
        action:0 up 1 down 2 left 3 right
        '''
        reward=-0.5
        done=False
        if action==0:
            self.agent_position[0]=max(0,self.agent_position[0]-1)
        elif action==1:
            self.agent_position[0]=min(self.size-1,self.agent_position[0]+1)
        elif action==2:
            self.agent_position[1]=max(0,self.agent_position[1]-1)
        elif action==3:
            self.agent_position[1]=min(self.size-1,self.agent_position[1]+1)
        else:
            raise ValueError
        if self.agent_position==self.bad1_position or self.agent_position==self.bad2_position:
            reward=-5
            done=True
        if self.agent_position==self.good_position:
            reward=5
            done=True
        return self.agent_position,reward,done

# if __name__=='__main__':
#     '''
#     Test Only
#     '''
#     env=Env()
#     env.render()
#     env.step(1)
#     env.render()
#     env.step(1)
#     env.render()

