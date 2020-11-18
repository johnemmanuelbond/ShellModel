# this class takes in a particle array and stitches together a movie of how it changes with time.
# the class has options to view the sedimentation down a particular axis (since the sedimentation code
# is fully 3D now)

from supports import *
from Collections import *
from Analyzer import *
from ParticleArray import *

class ArrayMovie:
    
    # initializes class instance, defines relevant variables
    def __init__(self, array, numsteps, stepsize, title = "Sedimentation Movie", axis = 'y',isIso=False):
        
        if(isinstance(array, ParticleArray) != True):
            raise Exception("Must input a ParticleArray instance")
        
        self.arr = array
        self.s = numsteps
        self.dt = stepsize
        self.N = self.arr.N
        
        self.ps = None
        self.os = None
        
        self.new = True
        
        if((axis=='x' or axis =='y' or axis=='z')!=True):
            raise Exception("Please correctly specifiy perspective (x y or z)")
        self.title = title

        self.axis = axis
        
        self.isIso = isIso

        self.cwd = os.getcwd()
    
    # runs the timestepping code 'steps' times and records the positions and orientations in an array
    def simulate(self, considerClumping=False):
        ind = np.arange(self.s)
        allps = np.zeros((self.s,self.N,3))
        allos = np.zeros((self.s,self.N,3))
        for i in ind:
            self.arr.formUp()
            allps[i] = self.arr.ps
            allos[i] = self.arr.os
            self.arr.timestep(self.dt,considerClumping=considerClumping)
        
        self.ps = allps
        self.os = allos
        
        self.new = False
    
    # gets the full 3d trajectories for each particle
    def getTrajectories(self):
        if(self.new):
            self.simulate()

        return self.ps, self.os

    # plots the rods and their orientations at a given step j
    def plotInstance(self, j, ax):
        # ensures sumulation is complete
        if(self.new):
            self.simulate()
        
        # determines viewing axis
        if(self.axis =='x'):
            xax=1
            yax=2
            oax=0
        elif(self.axis =='y'):
            xax=0
            yax=2
            oax=1
        elif(self.axis=='z'):
            xax=0
            yax=1
            oax=2
            
        # initializes matrices to hold data
        xdata = self.ps[:,:,xax]
        ydata = self.ps[:,:,yax]
        
        # ensures axis sizes are constant and big enough to view the simulation
        ax.set_xlim(np.min(xdata)-5, np.max(xdata)+5)
        ax.set_ylim(np.min(ydata)-5, np.max(ydata)+5)
        
        # sets axis labels and title
        ax.set_xlabel(np.array(['x','y','z'])[xax])
        ax.set_ylabel(np.array(['x','y','z'])[yax])
        plt.title(self.title)
        
        # plots the particles
        for i in np.arange(self.N):
            # gets the orientation and creates a marker at the correct angle
            if(self.isIso != True):
                o = self.os[j,i,oax]
                rod = mp.path.Path(np.array([[np.sin(o),np.cos(o)],[-np.sin(o),-np.cos(o)]]))
                ax.plot(self.ps[j,i,xax],self.ps[j,i,yax], linestyle='none', markeredgewidth = 2, marker = rod, markersize = 20);
            else:
                ax.plot(self.ps[j,i,xax],self.ps[j,i,yax], linestyle='none', marker = 'o', markeredgewidth = 2, markersize = 1);
    
    # plots the trajectories of the particles up until a specified step j
    def plotHistory(self, j, ax):
        # ensures sumulation is complete
        if(self.new):
            self.simulate()
        
        # determines viewing axis
        if(self.axis =='x'):
            xax=1
            yax=2
            oax=0
        elif(self.axis =='y'):
            xax=0
            yax=2
            oax=1
        elif(self.axis=='z'):
            xax=0
            yax=1
            oax=2
            
        # initializes matrices to hold data
        xdata = self.ps[:,:,xax]
        ydata = self.ps[:,:,yax]
        
        # ensures axis sizes are constant and big enough to view the simulation
        ax.set_xlim(np.min(xdata)-5, np.max(xdata)+5)
        ax.set_ylim(np.min(ydata)-5, np.max(ydata)+5)
        
        # sets axis labels and title
        ax.set_xlabel(np.array(['x','y','z'])[xax])
        ax.set_ylabel(np.array(['x','y','z'])[yax])
        plt.title(self.title)

        # plots said trajectories
        for i in np.arange(self.N):
            ax.plot(self.ps[0:j,i,xax],self.ps[0:j,i,yax], linewidth = 0.5);
    
    # this is a critical method, it updates an axis object to the current frame and has the right format to pass
    # into FuncAnimation. I learned to use the FuncAnimation method from: https://community.dur.ac.uk/joshua.borrow/blog/posts/making_research_movies_in_python/
    def updateAxes(self, frame, ax):
        ax.clear()
        self.plotInstance(frame, ax)
        self.plotHistory(frame, ax)
        
        return ax,
    
    # creates a movie in a certain location with a certain title
    def animate(self, title = "output", outputPath = None):
        fig, axes = plt.subplots();
        
        # see: https://community.dur.ac.uk/joshua.borrow/blog/posts/making_research_movies_in_python/
        # the exact specifications of some parameters are still a little arbitrary, specifically interval is free to change
        animation = mp.animation.FuncAnimation(fig, self.updateAxes, np.arange(self.s), fargs = [axes], interval = 100/self.dt)

        if(outputPath == None):
            outputPath = self.cwd+"\\outputMovies"

        if(os.path.isdir(outputPath)!=True):
            os.mkdir(outputPath)

        animation.save(outputPath + "\\" + self.title + ".mp4")

        plt.close(fig)

    #returns the plot at the final time.
    def getFinal(self):
        if(self.new):
            self.simulate()
        fig, ax = plt.subplots()

        self.updateAxes(self.s-1,ax)

        return fig

    # returns a plot with the particles at
    def getStill(self, times=10):
        if(self.new):
            self.simulate()
        fig, ax = plt.subplots()

        self.updateAxes(self.s-1,ax)
        for i in np.linspace(0,self.s,times,endpoint=False):
            self.plotInstance(i, ax)

        return fig