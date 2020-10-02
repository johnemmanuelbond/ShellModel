# code for performing the tirado calculation on a collection,
# this code is Now vectorized! (as of 9.6.2020).


from ShellModel.supports import *
from ShellModel.Collections import *


class Analyzer:

    # in otder to perform the shell model calculation we need a collection (col) and a tensor viscosity (eta).
    # there is also an option to have the principal axes of the tensor viscosity be rotated in relation to the
    # axes of the collection, the rotation axis is the x axis by default, finally limit controls for if the computer
    # running this calculation does not have enough memory to properly process a vectorized version of this code.
    def __init__(self, col, eta, theta=0, ax='x', limit=True):
        
        if(isinstance(col,Collection) != True):
            raise Exception("Must pass in a collection object as a parameter")
            
        self.col = col
        self.eta = rMat(eta,theta,axis=ax)
        self.T = None
        self.mat = None
        self.sol = None
        self.g = None
        self.Xi = None
        self.Thetas = None
        self.l=limit
    
    # calculates the hydrodynamic interaction tensor T
    def getT(self):
        col=self.col.copy()
        eta=np.copy(self.eta)
        
        N = self.col.N
        T = np.zeros((N,N,3,3))
        dR = np.copy(col.dR())
        
        # vectorized the looped code below
        
        i,j,a,b = np.mgrid[0:N,0:N,0:3,0:3]
        R = LA.norm(dR,axis=2)
        R[R==0]=-1
        T1 = np.zeros((N,N,3,3))
        T2 = np.zeros((N,N,3,3))
        
        T1[i,j,a,b] = (R[i,j]!=-1)*(1+(2*(col.a**2))/(3*(R[i,j]**2)))*np.eye(3)[a,b]/R[i,j]
        T2[i,j,a,b] = (R[i,j]!=-1)*(1-(2*(col.a**2))/(R[i,j]**2))*dR[i,j,a]*dR[i,j,b]/(R[i,j]**2)/R[i,j]
        T = 1/(8*np.pi)*LA.inv(eta) @ (T1+T2)
        
        #removing frivolous rounding errors
        T = T.round(8)
        
#         #loops through each pair of particles i,j and computes the hydrodynamic interaction tensor element using Tirado equation 2
#         for i in np.arange(N):
#             for j in np.arange(N):
#                 if (i != j):
#                     d = LA.norm(dR[i,j])
#                     prefactor = 1/(8*np.pi*d)
#                     c1 = 1 + (2*(col.a**2))/(3*(d**2))
#                     mat1 = np.eye(3)
#                     c2 = 1 - (2*(col.a**2))/(d**2)
#                     mat2 = np.outer(dR[i,j],dR[i,j])/(d**2)
#                     T[i,j] = LA.inv(eta) @ (prefactor*(c1*mat1+c2*mat2))
        
#         # removes frivolous precision errors
#         T = T.round(8)
        self.T = T
        return T
    
    #computes the drag tensor of each individual bead (they're all the same)
    def zeta(self):
        return 6*np.pi*self.eta*self.col.a
    
    # sets up the linear system needed to calculate the shielding tensors further down the road i.e. each side of 
    # Tirado equaiton 15. I've also used the a1 and b1 vectors to map the 3x3 shielding tensors G to 9-vectors, meaning
    # the linear equation consists of a 9N by 9N matrix and a 9N-vector which is the mapped identity
    def computeSystem(self):
        
        N = self.col.N
        z = self.zeta()
        
        #initialize the matrix and the other side of the equation
        mat = np.zeros((9*N,9*N))
        sol = np.zeros((9*N))
        
        # code to avoid repeating calculations if it's already been done
        if(self.T is None):
            T = self.getT()
        else:
            T = np.copy(self.T)
        
        # a1 and b1 contain the mapping information
        a1 = (np.floor(np.arange(9)/3)).astype(int)
        b1 = (np.arange(9)%3).astype(int)
        
        # useful indexing vectors, big index is the index of each G-component, gindex is the index within each G,
        # and particleindex is the index of each particle. This is necessary to keep track of which index of the 9N-vector 
        bigindex = (np.arange(9*N)).astype(int)
        gindex = (bigindex%9).astype(int)
        particleindex = (np.floor(bigindex/9)).astype(int)
        
        # ran into a memory problem for objects that are too big, 500 is an arbitrary cutoff.
        
        if((N >= 500) & self.l):
            # need to loop if the compound shape is too big\
            print("Vectorization will cause a memory problem. N = " + str(N))
            for i in bigindex:
                sol[i] = (a1[gindex[i]]==b1[gindex[i]])*1
                for j in bigindex:
                    A = z @ T[particleindex[i],particleindex[j]]
                    mat[i,j] = A[a1[gindex[i]],a1[gindex[j]]]*np.eye(3)[b1[gindex[j]],b1[gindex[i]]]
        else:
            # vectorized the looped code above
            sol[bigindex] = (a1[gindex[bigindex]]==b1[gindex[bigindex]])*1
            i,j = np.mgrid[0:9*N,0:9*N]
            A = np.zeros((9*N,9*N,3,3))
            A[i,j] = z@T[particleindex[i],particleindex[j]]
            mat[i,j] = A[i,j,a1[gindex[i]],a1[gindex[j]]]*np.eye(3)[b1[gindex[j]],b1[gindex[i]]]
        
        mat = mat + np.eye(9*self.col.N)
        mat = mat.round(8)
        
        self.mat = mat
        self.sol = sol
        return mat
    
    # solves the system of generated above for the shielding tensors
    def getShielding(self):
        # code to avoid repeating calculations if it's already been done
        if(self.mat is None):
            mat = self.computeSystem()
            sol = self.sol
        else:
            mat = self.mat
            sol = self.sol
        
        gvec = LA.solve(mat, sol)
        
        # reshapes gvec into a list of 3x3 tensors
        g = gvec.reshape(self.col.N,3,3)
        g = g.round(8)
        
        self.g = g
        return g
    
    # computes the drag tensor from the shielding tensors i.e. Tirado eqn 17, it's pretty easy
    def getXi(self, range = 'full'):
        # code to avoid repeating calculations if it's already been done
        if (self.g is None):
            g = self.getShielding()
        else:
            g = self.g
        
        z = self.zeta()
        Xis = np.zeros((self.col.N,3,3))
        ind = np.arange(self.col.N)
        
        Xis[ind] = z @ g[ind]
        
        if(range == 'full'):
            Xi = np.sum(Xis, axis=0)
        else:
            Xi = np.sum(Xis[range], axis=0)
        
        Xi = Xi.round(8)
        self.Xi=Xi
        return Xi
    
    # computes the rotational drag tensor (about axes through the coordinate cente) from the shielding tensors
    # I did this tensor math myself, when acting on an angular velocity, Theta gives the resulting torque due to drag
    # this bit of code is a much later addition to this class
    def getTheta(self, range = 'full'):
        # code to avoid repeating calculations if it's already been done
        if (self.g is None):
            g = self.getShielding()
        else:
            g = self.g
        
        # getting relevant matrices
        z = self.zeta()
        x = np.copy(self.col.pts)
        
        # using einsum method to calculate the complicated tensor math
        Thetas = np.einsum("ijk,aj,kl,alp,pqr,ar->aiq",LC,x,z,g,LC,x)
        
        if(range == 'full'):
            Theta = np.sum(Thetas, axis=0)
        else:
            Theta = np.sum(Thetas[range], axis=0)
        
        Theta = Theta.round(8)
        self.Theta=Theta
        return Theta
        