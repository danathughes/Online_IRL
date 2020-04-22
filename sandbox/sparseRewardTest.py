from environment import *
from reward import *
from sparse import *

stateSpace = DiscreteSpace(['a','b','c','d','e','f'])
actionSpace = DiscreteSpace(['w','x','y','z'])

parameters = np.array((-0.6,1.2,-0.2))


featureMap = FeatureMap(stateSpace,3)
featureMap.setFeature('a',1)
featureMap.setFeature('c',0)
featureMap.setFeature('c',2)
featureMap.setFeature('d',2)
featureMap.setFeature('f',0)


reward = LinearParametricReward(stateSpace, actionSpace, featureMap, parameters)

for state in stateSpace:
	print(state,':',reward(state,'w','a'))


sparseFeatureMap = SparseFeatureMap(stateSpace,3)
sparseFeatureMap.setFeature('a',1)
sparseFeatureMap.setFeature('c',0)
sparseFeatureMap.setFeature('c',2)
sparseFeatureMap.setFeature('d',2)
sparseFeatureMap.setFeature('f',0)


sparseReward = SparseLinearParametricReward(stateSpace, actionSpace, sparseFeatureMap, parameters)

for state in stateSpace:
	print(state,':', sparseReward(state,'w','a'))