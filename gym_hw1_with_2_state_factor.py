import gym
import numpy as np
import math
import random
env = gym.make("CartPole-v1")
#env.reset()

#a=(env.observation_space.sample())


#Use the pole angle and pole angle speed data
#Position and velocity of the cart is a little side objective, and can be solved by solving pole angle&angle velocity

ANGLE_DISCRETE_NO=5
ANGLE_SPEED_DISCRETE_NO=5
upperBounds=np.zeros(2)
lowerBounds=np.zeros(2)
upperBounds[0]=env.observation_space.high[2]
# upperBounds[0] = np.radians(360)
upperBounds[1]=math.radians(50)		
lowerBounds[0]=env.observation_space.low[2]
# lowerBounds[0] = -np.radians(360)
lowerBounds[1]=-math.radians(50)

QMAP=np.zeros(shape=(ANGLE_DISCRETE_NO,ANGLE_SPEED_DISCRETE_NO,2))
#ALPHA=0.01
#epsilon=0.01
gama=0.98

#Epsilon starts with 1 and drops to 0.1 with time
def chooseEpsilon(time):
	return 0.995 ** time
#Alpha starts with 1 and drops to 0.1 with time
def chooseAlpha(time):
        return 0.1
        # return max(0.1,min(1.0,1.0- math.log10((1+time)/10)))

#Discretize Pole Angle to [0,ANGLE_DISCRETE_NO-1]
def discretizePoleAngle(angle):
	angle= angle%(upperBounds[0]-lowerBounds[0])
	angle= angle/(upperBounds[0]-lowerBounds[0]) * ANGLE_DISCRETE_NO
	# print("Here is angle:", angle)
	return min(int(angle),ANGLE_DISCRETE_NO-1)
	#angle is [0,9] discrete

#Discretize Pole Angle Speed to [0,ANGLE_SPEED_DISCRETE_NO-1]
def discretizePoleAngleSpeed(speed):
	speed=speed%(upperBounds[1]-lowerBounds[1])
	speed= speed/(upperBounds[1]-lowerBounds[1]) * ANGLE_SPEED_DISCRETE_NO
	#print("Here is speed:",int(speed))
	##print(int(speed))
	return min(int(speed),ANGLE_SPEED_DISCRETE_NO-1)

def discretizeState(angle,speed):
	return (discretizePoleAngle(angle),discretizePoleAngleSpeed(speed),)

#Choose random action with probabilty epsilon
def chooseAction(epsilon,stateCo1,stateCo2):
	if np.random.random()<epsilon:
		return int(np.random.random()*2)
	else:
		return np.argmax(QMAP[stateCo1][stateCo2])
#Update QMAP
def updateQMAP(oldStateCo1,oldStateCo2,newStateCo1,newStateCo2,action,reward,alpha):
	change=alpha*(reward+gama*np.max(QMAP[newStateCo1][newStateCo2])-QMAP[oldStateCo1][oldStateCo2][action])
	QMAP[oldStateCo1][oldStateCo2][action]+=change
	#print("Change is %s",int(change))
#	QMAP[oldStateCo1][oldStateCo2][action]+= alpha*(reward+gama*np.argmax(QMAP[newStateCo1][newStateCo2])-QMAP[oldStateCo1][oldStateCo2][action])


oldObservation=env.reset()
oldStateCo1=discretizePoleAngle(oldObservation[2])
oldStateCo2=discretizePoleAngleSpeed(oldObservation[3])
#print (oldObservation)
accumulatedReward=0
for time in range(1000):
# you don’t have to render. it’s just for visualization.
	accumulatedReward=0
	alpha=chooseAlpha(time)
	epsilon=chooseEpsilon(time)
	oldObservation=env.reset()
	oldStateCo1=discretizePoleAngle(oldObservation[2])
	oldStateCo2=discretizePoleAngleSpeed(oldObservation[3])
	#print (oldObservation)
	for survival in range(500):
		#env.render()
		# take a random action
		##print(env.step(env.action_space.sample()))
		action=chooseAction(epsilon,oldStateCo1,oldStateCo2)
		#print("Action is:",action)
		newObservation, reward, done, info = env.step(action)
		accumulatedReward+=reward
		if done:
			print(time, "Survival tick is:",survival, epsilon)
			break
		
		newStateCo1=discretizePoleAngle(newObservation[2])
		newStateCo2=discretizePoleAngleSpeed(newObservation[3])
		#print("State is: %s, %s",newStateCo1,newStateCo2)
		updateQMAP(oldStateCo1,oldStateCo2,newStateCo1,newStateCo2,action,reward,alpha)
		#print(newObservation)
		##print(discretizePoleAngle(newObservation[2]))		
		##print(discretizePoleAngleSpeed(newObservation[3]))
				# oldObservation=newObservation
		oldStateCo2 = newStateCo2
		oldStateCo1 = newStateCo1
	# print(time, accumulatedReward)

print(QMAP)
env.close()
