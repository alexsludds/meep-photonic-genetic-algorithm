import matplotlib.pyplot as plt
import numpy as np
from main import upperPortGaussian, lowerPortGaussian

def generateGraph(photonicGrid, initialScore):
   plt.ion()
   plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=None)
   fig = plt.figure(1)
   ax = fig.add_subplot(311)
   myimshow = plt.imshow(photonicGrid) # Returns a tuple of line objects, thus the comma
   plt.colorbar()
   plt.title("Current Best Grid")
   plt.show(block = False)
   plt.pause(0.001)
   ax2 = fig.add_subplot(312)
   x = np.linspace(0,100,100)
   myimshow2, = plt.plot(x,initialScore)
   plt.colorbar()
   plt.title("Score Plot")
   plt.show(block=False)
   plt.pause(0.001)
   ax3 = fig.add_subplot(313)
   myimshow3 = plt.imshow(photonicGrid)
   plt.colorbar()
   plt.title("Current Array Being Tested")
   plt.show(block=False)
   plt.pause(0.001)
   ax4 = fig.add_subplot(321)
   t = np.linspace(1300,1750,100)
   ax4.plot(t,np.apply_along_axis(upperPortGaussian.computeGaussian,0,t),color='red')
   ax4.plot(t,np.apply_along_axis(lowerPortGaussian.computeGaussian,0,t),color='blue')
   plt.title("This is the score graph")
   plt.show(block=False)
   return myimshow, myimshow2, myimshow3, initialScore

def update_plot(plot, updateGrid):
    plot.set_data(updateGrid)
    plt.show(block=False)
    plt.pause(0.001)

def update_score(plot, updateScore, scoreArray):
    scoreArray = scoreArray[1:]
    plot.set_ydata(np.append(scoreArray,updateScore))
    plt.ylim(ymin=0,ymax=max(scoreArray))
    plt.show(block=False)
    plt.pause(0.001)

def update_testing_plot(plot,updateGrid):
    plot.set_data(updateGrid)
    plt.show(block=False)
    plt.pause(0.001)
