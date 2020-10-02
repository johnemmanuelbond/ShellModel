# this class creates an array of identical particles, given by a Collection 'particle', which start at
# initial positions 'posotions' with initial orientations 'orientations' subject to a fixed viscosity tensor and
# an external force


from ShellModel.supports import *
from ShellModel.Collections import *
from ShellModel.Analyzer import *

class ParticleArray:
    
    # initializing the object with error throwing for incorrect array dimensions
    def __init__(self, positions, particle, viscosity, fExt, orientations = 'default'):
        
        if(np.size(positions,1) != 3):
            raise Exception("Positions must be an array of 3D vectors")
        self.N = np.size(positions,0)
        self.pN = particle.N
        self.ps = positions
        self.ptcl = particle
        
        if (orientations == 'default'):
            orientations = np.zeros((positions.shape))
        if (positions.shape != orientations.shape):
            raise Exception("Position and orientation arrays need to be the same size")
        self.os = orientations
        
        if (np.size(viscosity,0) != 3 & np.size(viscosity,1) !=3):
            raise Exception("Viscosity must be 3x3 tensor")
        self.eta = viscosity
        
        if (np.size(fExt) != 3):
            raise Exception("External Force must be a 3D vector")
        self.f = fExt
        
        self.update = False
        
        self.gs = None
        self.z = None
        
        self.array = None
        self.Xis = None
        self.Thetas = None
        self.Nus = None
    
    #simple copy method
    def copy(self):
        return ParticleArray(np.copy(self.ps), self.ptcl.copy(), np.copy(self.eta), np.copy(self.f), orientations = np.copy(self.os))
    
    # produces one collection that is each particle located at it's position and orientation
    def formUp(self, vis = False, lim=True):
        self.update = True
        ind = np.arange(self.N-1)+1
        #need to initialize the first particle into the collection
        x = self.ptcl.copy()
        #important to rotate first, otherwise it will rotate about a non-central axis
        x.rotate(self.os[0,0],axis='x')
        x.rotate(self.os[0,1],axis='y')
        x.rotate(self.os[0,2],axis='z')
        x.translate(self.ps[0])
        arr = x
        # can do the rest in a loop
        for i in ind:
            x = self.ptcl.copy()
            x.rotate(self.os[i,0],axis='x')
            x.rotate(self.os[i,1],axis='y')
            x.rotate(self.os[i,2],axis='z')
            x.translate(self.ps[i])
            # adds each new particle to the large collection arr
            arr = merge(arr, x)
        self.array = arr
        if(vis):
            self.array.visualize()
        
        #computing shielding tensors
        an = Analyzer(arr, self.eta, limit=lim)
        z = an.zeta()
        gs = an.getShielding()
        
        self.gs = gs
        self.z = z
    
    # computes the translational drag tensor for each particle
    def getXis(self, lim=True):
        
        #making sure values are updated
        if(self.update == False):
            self.formUp(lim=lim)
        # initializing shielding tensors
        gs = self.gs
        z = self.z
        
        #initializing relevant indexing arrays
        bigindex = np.arange(self.N*self.pN)
        arrayindex = np.arange(self.N)
        constituentindex = np.arange(self.pN)
        
        
        # initializing arrays to hold the drag tensor for each individual constituent sphere as well as
        # the drag tensors for each particle
        xis = np.zeros((self.N*self.pN,3,3))
        Xis = np.zeros((self.N,3,3))
        
        #computes the drag tensors on each individual sphere
        xis[bigindex] = z @ gs[bigindex]
        
        # the drag coeffecient for each particle is a sum over Xis for the constituent spheres that make up each particle.
        # since the particles are added in order, this sum is over a 
        
        for i in arrayindex:
            Xis[i] = np.sum(xis[self.pN*i+constituentindex], axis=0)
        
        Xis = Xis.round(8)
        self.Xis = Xis
        
        return Xis
    
    # computes the rotational drag tensor for each particle about it's center
    def getThetas(self, lim=True):
        
        #making sure values are updated
        if(self.update == False):
            self.formUp(lim=lim)
        # initializing shielding tensors
        gs = self.gs
        z = self.z
        
        #initializing relevant indexing arrays
        bigindex = np.arange(self.N*self.pN)
        arrayindex = np.arange(self.N)
        constituentindex = np.arange(self.pN)
        
        #getting positions off-center for each particle
        p = np.zeros((self.N*self.pN,3))
        pts = np.copy(self.array.pts)
        for i in arrayindex:
            p[self.pN*i+constituentindex] = pts[self.pN*i+constituentindex] - self.ps[i]
        
        
        # initializing arrays to hold rotaitonal the drag tensors for each constituent
        Thetas = np.zeros((self.N,3,3))
        
        #computes the rotational drag tensors on each individual sphere
        ts = np.einsum("ijk,aj,kl,alp,pqr,ar->aiq",LC,p,z,gs,LC,p)
        
        #adding together the rotational drag tensors over the constituents
        for i in arrayindex:
            Thetas[i] = np.sum(ts[self.pN*i+constituentindex], axis=0)
        
        Thetas = Thetas.round(8)
        self.Thetas = Thetas
        
        return Thetas
    
    # returns a tensor which, when acting on a net velocity, gives the resultant torque about the center.
    # This effect only manifests because the presence of other particles in the array disrupts the symmetry
    # of the drag tensors througout the extent of one particle, so a net velocity of all the constituents
    # can produce a net torque.
    def getNus(self, lim=True):
        
        #making sure values are updated
        if(self.update == False):
            self.formUp(lim=lim)
        # initializing shielding tensors
        gs = self.gs
        z = self.z
        
        #initializing relevant indexing arrays
        bigindex = np.arange(self.N*self.pN)
        arrayindex = np.arange(self.N)
        constituentindex = np.arange(self.pN)
        
        #getting positions off-center for each constituent
        p = np.zeros((self.N*self.pN,3))
        pts = np.copy(self.array.pts)
        for i in arrayindex:
            p[self.pN*i+constituentindex] = pts[self.pN*i+constituentindex] - self.ps[i]
        
        # initializing array to hold these tensors
        Nus = np.zeros((self.N,3,3))
        
        # using einsum to calculate the tensors
        nus = np.einsum("ijk,aj,kp,apl->ail",LC,p,z,gs)
        
        # summing over each constituent to get the tensor for each particle
        for i in arrayindex:
            Nus[i] = np.sum(nus[self.pN*i+constituentindex], axis=0)
        
        Nus = Nus.round(8)
        self.Nus = Nus
        
        return Nus
    
    # this code steps forward the position and oridentations in time.
    def timestep(self, dt, considerClumping = False):
        Xis = self.getXis();
        Thetas = self.getThetas();
        Nus = self.getNus();
        N=self.N;
        pN=self.pN;
        
        # initializng arrays for indexing through the particles and storing their instantaneous velocoties
        vs = np.zeros((self.N,3))
        ws = np.zeros((self.N,3))
        ind = np.arange(self.N)
        
        # calculating instantaneous velocoties (assuming low Reynolds number)
        for i in ind:
            vs[i] = LA.pinv(-Xis[i])@(-self.f)
            ws[i] = LA.pinv(-Thetas[i])@(Nus[i]@vs[i])
        
        #deals with clumping if that's a consideration
        if(considerClumping):
            # gets the array of displacement vectors between constituents and determines which ones are close
            # enough to clump
            dR = self.array.dR().round(8)
            close = (LA.norm(dR,axis=2)<2*self.ptcl.a)*1-np.eye(N*pN)
            
            # notes which particles are clumping
            i,j = np.mgrid[0:pN*N,0:pN*N]
            touching=np.zeros((N,N))
            touching[(i/pN).astype(int),(j/pN).astype(int)] = close[i,j].astype(int)
            
            # forms a list of pairs of clumped particles
            i,j = np.mgrid[0:N,0:N]
            touching[i>j]=0
            i,j = np.where(touching)
            pairs = np.transpose(np.array([i,j]))
            
            # assigns particles new ws and vs according to clumping rules
            # 1) clumped particles move with the same average velocity
            # 2) clumped particles rotate so that they are not crossing (this code can probably be refined)
            # this code cannot yet deal with clumps of more than two particles
            for pair in pairs:
                vs[pair] = np.mean(vs[pair],axis=0)
                ws[pair] = 1/dt*np.outer(np.array([1/2,-1/2]),np.diff(self.os[pair],axis=0)[0])#np.mean(ws[pair],axis=0)
        
        # stepping forward in time using x(t+dt) = x(t) + v(t)dt
        newPs = self.ps + vs*dt
        newOs = self.os + ws*dt
        
        # updates positions and orientations

        self.ps = newPs
        self.os = newOs
        
        self.update = False
