import matplotlib.pyplot as plt
import numpy as np

class FindCenterPointState():
	def __init__(self, background):
		self.background = background

	def __call__(self, grid):
		xbackground = self.background[0][1]-self.background[0][0]
		ybackground = self.background[1][1]-self.background[1][0]
		xGridLine = [self.background[0][0]+i*xbackground/(grid[0]) for i in range(0,grid[0]+1)]
		yGridLine = [self.background[1][0]+i*ybackground/(grid[1]) for i in range(0,grid[1]+1)]
		xCenter = [(xGridLine[i] + xGridLine[i+1])/2 for i in range(0,grid[0])]
		yCenter = [(yGridLine[i] + yGridLine[i+1])/2 for i in range(0,grid[1])]
		return xGridLine, yGridLine, xCenter, yCenter

class FindCenterPointValue():
	def __init__(self, valueFunction):
		self.valueFunction = valueFunction

	def __call__(self, xCenter, yCenter):
		centerPointValue = []
		for y in yCenter:
		     centerPointValue.append([self.valueFunction([x,y]) for x in xCenter])
		return centerPointValue
		

class DrawValueMap():
	def __init__(self, findCenterPointState, findCenterPointValue, background):
		self.findCenterPointValue = findCenterPointValue
		self.findCenterPointState = findCenterPointState
		self.background = background#delete

	def __call__(self, grid):
		xGridLine, yGridLine, xCenter, yCenter = self.findCenterPointState(grid)
		centerPointValue = self.findCenterPointValue(xCenter, yCenter)

		x, y = np.meshgrid(xGridLine, yGridLine)
		value = np.array(centerPointValue)
		plt.pcolormesh(x, y, value)
		plt.colorbar()
		plt.show()

