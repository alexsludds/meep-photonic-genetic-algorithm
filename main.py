import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import os , sys, time
import subprocess
from sqlalchemy import *
from sqlalchemy.engine.url import URL
from sqlalchemy.types import TIMESTAMP
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Float, Boolean, desc
from datetime import datetime
from sqlalchemy.ext.declarative import declarative_base
import graph


class grid():
        def __init__(self):
            self.xPixels = 100
            self.yPixels = 100
            self.xPixelDimension = 1
            self.yPixelDimension = 1
            self.xLength = self.xPixels * self.xPixelDimension
            self.yLength = self.yPixels * self.yPixelDimension

        def getXPixels(self):
            return self.xPixels
class cell():
        def __init__(self):
            self.x = 1
            self.y = 1

class upperPort():
    gridInstance = grid()
    x = gridInstance.xLength/2
    y = gridInstance.yLength/2

class lowerPort():
    gridInstance = grid()
    x = gridInstance.xLength/2
    y = -1*gridInstance.yLength/2

class block():
        def __init__(self):
            self.epsilon = 11.68 #This is the permitivity of silicon used for integrated devices

class source():
        def __init__(self):
            self.beginningFrequency  = 0.15
            self.endFrequency = 0.20
            self.x = 0
            self.y = 0

class dbMeta():
        def __init__(self):
            self.sqliteURI = 'sqlite:///test.db'

        def getsqliteURI(self):
                return self.sqliteURI

class upperPortGaussian():
    def __init__(self):
        self.sigma = 50.0
        self.mu = 1350 #wavelength in nanometers

    def computeGaussian(self,x):
        return 1.0/np.sqrt(2*np.pi*self.sigma**2.0)*np.exp(-0.5*((x-self.mu)/self.sigma)**2.0)

    def mu(self):
        return self.mu


class lowerPortGaussian():
    def __init__(self):
        self.sigma = 50.0
        self.mu = 1700 #wavelength in nanometers

    def computeGaussian(self,x):
        return 1.0/np.sqrt(2*np.pi*self.sigma**2.0)*np.exp(-0.5*((x-self.mu)/self.sigma)**2.0)

    def mu(self):
        return self.mu

class simulation():
    def __init__(self):
        self.simulationTimeSteps = 200
        self.pngAtEvery = self.simulationTimeSteps/10
        self.resolution = 10
        self.numberOfFrequenciesToSimulate = 20
        self.initialTemperature = 5000
        self.finalTemperatureCutoff = 5
        self.generatePNG = False
        self.simulationPNGSettings = "-Zc dkbluered -a yarg -A main-out/main-eps-000000.00.h5 main-out/main-ez.h5"




grid = grid()
cell = cell()
upperPort = upperPort()
lowerPort = lowerPort()
upperPortGaussian = upperPortGaussian()
lowerPortGaussian = lowerPortGaussian()
block = block()
source = source()
dbMeta = dbMeta()
simulation = simulation()

Base = declarative_base()

engine = create_engine(dbMeta.sqliteURI, echo=True)
Session = sessionmaker(bind=engine)
session = Session()


class arrayStorage(Base):
    __tablename__ = "arrayStorage"
    id = Column(Integer, primary_key=True,unique=True)
    arrayString = Column(String)
    timeAdded = Column(TIMESTAMP, default=datetime.now)
    score = Column(Float)

    def __repr__(self):
        return "id: %s, arrayString: %s, timeAdded: %s, score: %s" % (self.id, arrayStorage.stringToNumpyArray(self.arrayString), self.timeAdded, self.score)

    def numpyArrayToString(numpyArray):
        print(numpyArray.tostring(), "THIS IS A NUMPY ARRAY TURNED INTO A STRING")
        numpyArrayAsString = numpyArray.tostring()
        return numpyArrayAsString

    def stringToNumpyArray(string):
        byte_array = np.fromstring(string , dtype=int)
        #Convert the array from a 1D array into a 2D array
        byte_array_2d = np.reshape(byte_array, (grid.xLength,grid.yLength))
        print("Numpy array has been reshaped")
        return byte_array_2d

    def returnMostRecentNumpyArray():
        queryID = session.query(arrayStorage).order_by(arrayStorage.timeAdded.desc()).first().id
        queryArrayString = session.query(arrayStorage).filter_by(id=queryID).first().arrayString
        numpyArray = arrayStorage.stringToNumpyArray(queryArrayString)
        return numpyArray

    def returnBestCost():
        best_cost_entry = session.query(arrayStorage).order_by(arrayStorage.score.desc()).first()
        print(best_cost_entry)
        return best_cost_entry.score

    def numberOfEntriesInDatabase():
        return session.query(arrayStorage.id).count()

    def addNumpyArray(numpyArray,cost):
        newArray = arrayStorage(arrayString= arrayStorage.numpyArrayToString(numpyArray), score = cost)
        session.add(newArray)
        session.commit()

def db_connect():
    return create_engine(URL(**settings.database))

def createPNG():
    print("Creating PNG")
    #fileOS = os.popen("h5ls main-out/main-ez.h5").stdout.read()
    pngString = "h5topng -t 0:" + str(simulation.simulationTimeSteps-1) + " -R -Zc dkbluered -a yarg -A main-out/main-eps-000000.00.h5 main-out/main-ez.h5"
    # pngString = "h5topng -t 0:199 -R -Zc dkbluered -a yarg -A main-out/main-eps-000000.00.h5 main-out/main-ez.h5"

    os.system(pngString)

def pngToGIF():
    print("Converting PNG to GIF")
    gifCreationString = "convert main-out/main-ez.t*.png main-ez.gif"
    os.system(gifCreationString)
    print("Finished Creating GIF")
    os.system("xdg-open main-ez.gif")

def deleteH5Files():
    print("Deleting H5 Files")
    h5DeletionString = "rm main-out/main-eps-000000.00.h5 main-out/main-ez.h5"
    os.system(h5DeletionString)

def initializePhotonicGrid():
    photonicGrid = np.zeros((grid.xLength,grid.yLength))
    return photonicGrid

def randomPhotonicGrid():
    photonicGrid = np.random.choice([0, 1], size=(grid.xLength,grid.yLength), p=[0.5,0.5])
    return photonicGrid

def blockIndexToCenter(xindex,yindex):
    x = grid.xPixelDimension/2.0 + xindex * grid.xPixelDimension - grid.xLength/2
    y = grid.yPixelDimension/2.0 + yindex * grid.yPixelDimension - grid.yLength/2
    return x , y

def createBlockObject(x,y,epsilon=block.epsilon):
    blockObject = mp.Block(mp.Vector3(grid.xPixelDimension,grid.yPixelDimension,0),
             center=mp.Vector3(x,y,0),
             material = mp.Medium(epsilon=1))
    return blockObject

def generatePhotonicGrid(numpyArray):
    photonicArray = numpyArray
    geometry = [mp.Block(mp.Vector3(grid.xLength, grid.yLength, 1),
                     center=mp.Vector3(0, 0, 0),
                         material=mp.Medium(epsilon=1))]
    areasWithSilicon = np.argwhere(photonicArray == 1)
    for i in areasWithSilicon:
        xindex = i[0]
        yindex = i[1]
        x , y = blockIndexToCenter(xindex,yindex)
        block = createBlockObject(x,y)
        geometry.append(block)
    return geometry

def createLeftSideWaveguide():
    pass

def volumeMeasureFromUpperPort():
    volume = mp.Volume(center = mp.Vector3(upperPort.x , upperPort.y , 0) ,
                           size = mp.Vector3(grid.xPixelDimension , grid.yPixelDimension))
    return volume

def volumeMeasureFromLowerPort():
    volume = mp.Volume(center = mp.Vector3(lowerPort.x , lowerPort.y , 0) ,
                           size = mp.Vector3(grid.xPixelDimension , grid.yPixelDimension))
    return volume

'''
We want 1350nm light to come out of the upper port
We want 1700nm light to come out of the lower port
'''
def successAtFrequency(upperPortMeasure , lowerPortMeasure, wavelengthOfMeasurement):
    #we are going to use two gaussians, each centered around the frequencies of interest.
    return upperPortGaussian.computeGaussian(wavelengthOfMeasurement)*upperPortMeasure + lowerPortGaussian.computeGaussian(wavelengthOfMeasurement)*lowerPortMeasure

def runSimulation(cell,geometry,pml_layers,frequency):
    print("We are running at:", frequency)
    resolution = simulation.resolution
    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        resolution = resolution,
                        progress_interval = 10000,
                        sources = [mp.Source(mp.ContinuousSource(frequency = frequency),
                             center = mp.Vector3(source.x , source.y , 0),
                                             component=mp.Ex)])
    sim.use_output_directory()
    if(simulation.generatePNG==True):
            sim.run(
                    mp.to_appended("ez", mp.at_every(1.0, mp.output_efield_z)),
                    until=simulation.simulationTimeSteps)
            print("After Simulation")
            # createPNG()
            # pngToGIF()
            #deleteH5Files()
    else:
            sim.run(
                    until=simulation.simulationTimeSteps)
    upperPortVolume = mp.Volume(center = mp.Vector3(upperPort.x,upperPort.y,0), size = mp.Vector3(1,1,0))
    lowerPortVolume = mp.Volume(center = mp.Vector3(lowerPort.x,lowerPort.y,0), size = mp.Vector3(1,1,0))
    lowerPortEnergy = sim.field_energy_in_box(box = lowerPortVolume, d=mp.Ex)
    upperPortEnergy = sim.field_energy_in_box(box = upperPortVolume, d=mp.Ex)
    print(lowerPortEnergy, "lowerPortEnergy")
    print(upperPortEnergy, "upperPortEnergy")
    return lowerPortEnergy,upperPortEnergy

def wavelengthsInNMToTest():
    return np.linspace(upperPortGaussian.mu,lowerPortGaussian.mu,simulation.numberOfFrequenciesToSimulate)

def frequencyToWavelengthInNM(freq):
    return 1.0/(freq*1000)

def wavelengthInNMToFrequency(wavelength):
    return 1.0/(wavelength*1000)

def calculateCost(cell,geometryAsNumpyArray,pml_layers):
    testingWavelengths = wavelengthsInNMToTest()
    testingFrequencies = [wavelengthInNMToFrequency(x) for x in testingWavelengths]
    totalcost = 0
    geometry = generatePhotonicGrid(geometryAsNumpyArray)
    for index,testFrequency in enumerate(testingFrequencies):
        lowerPortEnergy, upperPortEnergy = runSimulation(cell,geometry,pml_layers, testFrequency)
        wavelength = testingWavelengths[index]
        cost = successAtFrequency(upperPortEnergy,lowerPortEnergy,wavelength)
        totalcost += cost
    return totalcost

def simulatedAnnealingAlgorithm(cell,pml_layers):
    plotArray, plotScore, plotTesting, scoreArray = graph.generateGraph(np.zeros((grid.xPixels,grid.yPixels)),initialScore = np.zeros(100))
    testingWavelengths = wavelengthsInNMToTest()
    testingFrequencies = [wavelengthInNMToFrequency(x) for x in testingWavelengths]
    print(arrayStorage.numberOfEntriesInDatabase())
    if(arrayStorage.numberOfEntriesInDatabase()==0):
        #generate a random solution
        photonicGrid = initializePhotonicGrid()
        #calculate cost using cost function
        initialcost = calculateCost(cell,photonicGrid,pml_layers)
        arrayStorage.addNumpyArray(photonicGrid,initialcost)

    temperature = simulation.initialTemperature
    while temperature > simulation.finalTemperatureCutoff:
        #generate a random neighboring solution
        currentBestArray = arrayStorage.returnMostRecentNumpyArray()
        if(temperature % 10 == 0):
                currentBestArray = randomPhotonicGrid()
        x = np.random.randint(0,grid.xPixels)
        y = np.random.randint(0,grid.yPixels)
        if(currentBestArray[x,y]==0):
            testArray = currentBestArray
            testArray[x,y] = 1
        elif(currentBestArray[x,y]==1):
            testArray = currentBestArray
            testArray[x,y] = 0
        else:
            print("Error occured, array element was not 0 or 1")
        graph.update_testing_plot(plotTesting,testArray)
        testCost = calculateCost(cell,testArray,pml_layers)
        print("The test cost is ",testCost)
        #compare
        if(testCost > arrayStorage.returnBestCost()):
            arrayStorage.addNumpyArray(testArray,testCost)
            graph.update_plot(plotArray,testArray)
            graph.update_score(plotScore,testCost,scoreArray)

#plotimshow, plotimshow2 , scoreArray = graph.generateGraph(np.random.rand(10,10),0)

if __name__ == "__main__":
    Base.metadata.create_all(bind=engine)
    simulatedAnnealingAlgorithm(cell = mp.Vector3(grid.xLength , grid.yLength , 0),
                            pml_layers = [mp.PML(1.0)])
    # cell = mp.Vector3(grid.xLength,grid.yLength,0)
    # pml = [mp.PML(1.0)]
    # arrayphoto = np.random.randint(2,size=(20,20))
    # geometry = generatePhotonicGrid(arrayphoto)
    # runSimulation(cell=cell,geometry=geometry,pml_layers=pml,frequency=0.4)
