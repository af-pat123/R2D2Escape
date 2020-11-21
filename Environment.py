import pybullet as p 
import pybullet_data 
import time 
import numpy as np 
from math import sin, cos
import cv2
import matplotlib.pyplot as plt
import pygame


#TODO Take action for K timesteps, and sample frame after K timesteps Zarar
#TODO Run multiple envs in the world Afzal Rajat Yousef Zarar
#TODO Fix getObservation func Rajat


class Environment(object):

    '''
    State Space: (width, height, frameStackSize)

    Action Space:

    0 - w - move forward
    1 - s - move backwards
    2 - a - rotate counterclockwise
    3 - d - rotate clockwise
    
    potential actions

    wa - 
    wd - 
    sa -
    sd -  

    '''

    def __init__(self):

        
        client = p.connect(p.GUI) 
        p.setTimeOut(2)
        p.setGravity(0,0,-9.8)
        self.speed = 20
        self.walls = []
        self.wallThickness = 0.1
        self.wallHeight = 1 
        self.wallColor = [1, 1, 1, 1]
        self.agent = 'r2d2.urdf'
        self.rotationSpeed = 20
        self.max_timesteps = 10000
        self.spawnPos = [0, 0, 1]
        self.spawnOrn = p.getQuaternionFromEuler([0, 0, 0])
        self.prevAction = -1
        self.forces = 100
        self.frameStackSize = 4
        self.frames = []
        self.collisionDetected = False
        self.counter = 0 
        self.K = 3
        self.viewer = None 
        self.imgHeight = self.imgWidth = 300
        self.screen = None
        
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())



    def generate_world(self, agent='r2d2.urdf', escapeLength=50, corridorLength= 5,numObstacles=10, obstacleOpeningLength=0.5,  r2d2DistanceAheadOfWall=3, seed=42):
        
        totalLength = escapeLength + r2d2DistanceAheadOfWall
        np.random.seed(seed)
        distanceBetweenObstacles = escapeLength/numObstacles

        p.resetSimulation()
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")       
        self.r2d2Id = p.loadURDF(agent, self.spawnPos, self.spawnOrn)

        # Create walls enclosing tunnel
 
        self.nsHalfExtents = [corridorLength, self.wallThickness, self.wallHeight]
        self.ewHalfExtents = [self.wallThickness, totalLength/2, self.wallHeight]
        
        self.nWallCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=self.nsHalfExtents)
        self.sWallCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=self.nsHalfExtents)
        self.eWallCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=self.ewHalfExtents)
        self.wWallCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=self.ewHalfExtents)

        self.nWallVisualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, rgbaColor=self.wallColor, halfExtents=self.nsHalfExtents)
        self.sWallVisualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, rgbaColor=self.wallColor, halfExtents=self.nsHalfExtents)
        self.eWallVisualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, rgbaColor=self.wallColor, halfExtents=self.ewHalfExtents)
        self.wWallVisualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, rgbaColor=self.wallColor, halfExtents=self.ewHalfExtents)
 
        self.nWallId = p.createMultiBody(basePosition=[0, escapeLength, self.wallHeight], baseCollisionShapeIndex=self.nWallCollisionShapeId, baseVisualShapeIndex=self.nWallVisualShapeId) 
        self.sWallId = p.createMultiBody(basePosition=[0, -r2d2DistanceAheadOfWall, self.wallHeight], baseCollisionShapeIndex=self.sWallCollisionShapeId, baseVisualShapeIndex=self.sWallVisualShapeId)
        self.eWallId = p.createMultiBody(basePosition=[corridorLength, -r2d2DistanceAheadOfWall/2+escapeLength/2, self.wallHeight], baseCollisionShapeIndex=self.eWallCollisionShapeId, baseVisualShapeIndex=self.eWallVisualShapeId) 
        self.wWallId = p.createMultiBody(basePosition=[-corridorLength, -r2d2DistanceAheadOfWall/2+escapeLength/2,self.wallHeight], baseCollisionShapeIndex=self.wWallCollisionShapeId, baseVisualShapeIndex=self.wWallVisualShapeId)


        self.walls.extend([self.nWallId, self.sWallId, self.eWallId, self.wWallId])

        # Generate obstacles

        for i in range(numObstacles):
            
            obstacleYCord = distanceBetweenObstacles*(i) + r2d2DistanceAheadOfWall
            max_length = corridorLength - obstacleOpeningLength
            westX = np.random.rand()*max_length 
            eastX = corridorLength - westX - obstacleOpeningLength
            westWallCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[westX, self.wallThickness, self.wallHeight])
            westWallVisualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, rgbaColor=self.wallColor, halfExtents=[westX, self.wallThickness, self.wallHeight])
            
            eastWallCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[eastX, self.wallThickness, self.wallHeight])
            eastWallVisualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, rgbaColor=self.wallColor, halfExtents=[eastX, self.wallThickness, self.wallHeight])

            eastWallBasePosition = [corridorLength-self.wallThickness-eastX, obstacleYCord, self.wallHeight]
            westWallBasePosition = [-corridorLength+westX, obstacleYCord, self.wallHeight]
            
            wObstacleId = p.createMultiBody(baseCollisionShapeIndex=eastWallCollisionShapeId, baseVisualShapeIndex=eastWallVisualShapeId, basePosition=eastWallBasePosition)
            eObstacleId = p.createMultiBody(baseCollisionShapeIndex=westWallCollisionShapeId, baseVisualShapeIndex=westWallVisualShapeId, basePosition=westWallBasePosition)

            self.walls.extend([wObstacleId, eObstacleId])



    def reset(self, agent='r2d2.urdf'):
        '''
        Reset world state to given initial state
        '''

        self.timestep = 0
        self.collisionDetected = False
        self.generate_world()

        for _ in range(100):
            p.stepSimulation()
            self.frames.append(self.getFrame())
            self.timestep +=1 

        self.frames = self.frames[-self.frameStackSize:]

        initialObs = self.getObservation()

        return initialObs

    def step(self, action):

        pos_t, orn_t = p.getBasePositionAndOrientation(self.r2d2Id)

        pos_t1 = pos_t = np.array(pos_t)
        orn_t1 = orn_t = np.array(p.getEulerFromQuaternion(orn_t))

        if action != self.prevAction:
            prevAction = action
            self.setAction(action)

        p.stepSimulation()

        self.frames.append(self.getFrame())
        self.frames = self.frames[-self.frameStackSize:]

        nextObs = self.getObservation()
        done = self.isDone()
        reward = self.getReward()
        
        debug = []

        self.timestep += 1

        return nextObs, reward, done, debug

    def setAction(self,action):
        '''
        Sets the action the r2d2 should be taking (move forward/backward or rotate CW, CCW).
        There are 15 joints in this robot.

        Joint at index 2 is "right_front_wheel_joint"
        Joint at index 3 is "right_back_wheel_joint"
        Joint at index 7 is "left_front_wheel_joint"
        Joint at index 8 is "left_back_wheel_joint"

        These are the joints that matter for movement. Following code can give information of all joints:

        for i in range(0,15):
            print(p.getJointInfo(self.r2d2Id,i))
        
        Next Steps: Find a mechanical engineer to setup specific values of the relevant joints, to obtain refined movement.
        '''

        if action == 0:
                p.setJointMotorControlArray(bodyUniqueId = self.r2d2Id,
                                            jointIndices = [2,3,6,7], 
                                            controlMode = p.VELOCITY_CONTROL,
                                            targetVelocities = [-20,-20,-20,-20],
                                            forces = [100,100,100,100])
        elif action == 1:
                p.setJointMotorControlArray(bodyUniqueId = self.r2d2Id,
                                            jointIndices = [2,3,6,7], 
                                            controlMode = p.VELOCITY_CONTROL,
                                            targetVelocities = [20,20,20,20],
                                            forces = [100,100,100,100])
        elif action == 2:
                p.setJointMotorControlArray(bodyUniqueId = self.r2d2Id,
                                            jointIndices = [3,6], 
                                            controlMode = p.VELOCITY_CONTROL,
                                            targetVelocities = [80,-80],
                                            forces = [100,100])
        elif action == 3:
            p.setJointMotorControlArray(bodyUniqueId = self.r2d2Id,
                                            jointIndices = [2,7], 
                                            controlMode = p.VELOCITY_CONTROL,
                                            targetVelocities = [-80,80],
                                            forces = [100,100])
    
    def getObservationZ(self):
        '''
        Input list(4, height, width) -> Output NpArray(height, width, 4)
        Return stack of frames as numpy array of shape (width, height, stackSize) also normalized Rajat
        '''

        frames = self.frames[-self.frameStackSize:]
        obs = np.zeros((self.imgWidth, self.imgHeight, self.frameStackSize))
        
        for i, frame in enumerate(frames):
            obs[:,:,i] = frame

        return obs


    def getObservation(self):
        '''
        Input list(num_frames, w, h, rgba) -> Output NpArray(w, h, num_frames)
        '''
        
        #frames = self.frames[-self.frameStackSize:]                                             
        frames = np.transpose(self.frames,(1,2,0))                           
        frames /= 255.0  
            
        return frames
    
    def getFrame(self):
        '''
        Returns the observation 
        '''
    
        pos, orn = p.getBasePositionAndOrientation(self.r2d2Id) 

        eyePos = [pos[0],pos[1],1.5]
        
        orn = np.array(p.getEulerFromQuaternion(orn))  
        yaw=orn[2]+3.14159/2
        targPos = [2*cos(yaw)+pos[0],2*sin(yaw)+pos[1],1]
     
        viewMatrix = p.computeViewMatrix(cameraEyePosition=eyePos,cameraTargetPosition=targPos,cameraUpVector=[0,0,1])  

        projectionMatrix = [
        1.0825318098068237, 0.0, 0.0, 0.0, 0.0, 1.732050895690918, 0.0, 0.0, 0.0, 0.0,
        -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0]
        frame = p.getCameraImage(self.imgWidth, self.imgHeight, viewMatrix=viewMatrix, projectionMatrix=projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)[2]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)/255.0

        return frame 


    def getReward(self, weight=1):
        '''
        Calculates and returns the reward that the agent maximizes
        '''
        if not self.collisionDetected:
            pos, orn = p.getBasePositionAndOrientation(self.r2d2Id)
            reward = pos[0] #- self.timestep*(weight)
        else:
            reward = -1

        return reward 


    def isDone(self):
        '''
        Returns True if agent completes escape (x >= 100) or if episode duration > max_episode_length or if collision is detected 
        '''
        contact = False
        pos, orn = p.getBasePositionAndOrientation(self.r2d2Id)
        

        try:
            contactPoints = np.array(p.getContactPoints(self.r2d2Id))
            contactBodyIds = contactPoints[:,2]
            contact = any(body in contactBodyIds for body in self.walls)
    
        except IndexError as e:
            print(e)

        if contact:
            self.collisionDetected = True

        if (pos[1] >= 100) | (self.timestep>=self.max_timesteps) | self.collisionDetected:
            done=True
        else:
            done=False
        
        return done     

    def render(self):

        '''
        Instead of rendering the physics engine by using a p.GUI connection method, we are going to connect using p.DIRECT and 
        render using cameras instead. 
        #TODO Create a 3rd person view camera and make its position relative to the agent's position. This camera is different 
        than the one used to get the observation for the agent. 
        #TODO Grab frames using this camera and draw the images to a window using pygame.        
        #TODO Have the option to use different camera angles for viewing 
        '''

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.imgWidth, self.imgHeight))
        
        frame = self.frames[-1]

        frameSurface = pygame.surfarray.make_surface(frame)
        frameSurface = pygame.transform.rotate(frameSurface, -90)
        self.screen.blit(frameSurface, (0, 0))
        pygame.display.update()

        


