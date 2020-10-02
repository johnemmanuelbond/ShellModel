#imports


from supports import*

#code for representing collections of beads, it needs to store a series of vectors
# as well as the radius of each bead.

class Collection:
    
    def __init__(self,pts,a=1):
        self.a = a
        self.pts = pts
        self.N = np.size(self.pts,0)
    
    #adds a point to the collection
    def add(self,point):
        p = np.copy(self.pts)
        self.pts = np.vstack((p,point))
        self.N = np.size(self.pts,0)
    
    #rotates the collection along a specified axis and angle
    def rotate(self, theta, axis='x'):
        p = np.transpose(np.copy(self.pts))
        if(axis == 'x' or axis == 1):
            self.pts = np.transpose(rx(theta) @ p)
        if(axis == 'y' or axis == 2):
            self.pts = np.transpose(ry(theta) @ p)
        if(axis == 'z' or axis == 3):
            self.pts = np.transpose(rz(theta) @ p)
    
    #translates the collection along a vector
    def translate(self, vector):
        p = np.copy(self.pts) + vector
        self.pts = p
    
    def dR(self):
        i,j = np.mgrid[0:self.N,0:self.N]
        p = np.copy(self.pts)
        return p[i] - p[j]
    
    #plots the collection
    def visualize(self):
        p = np.copy(self.pts)
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(p[::,0],p[::,1],p[::,2],s=30*np.pi*(self.a**2))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        axisEqual3D(ax)
    
    #creates a copy of the collection
    def copy(self):
        return Collection(self.pts,a=self.a)

#creates a sphere of beads
class Sphere(Collection):
    
    def __init__(self,Rrel,a=1):
        self.a = a
        self.Rrel = Rrel
        
        #generates a close-packed grid of points that contains a sphere
        a1 = np.sqrt(2)*np.array([1/2,1/2,0])
        a2 = np.sqrt(2)*np.array([1/2,0,1/2])
        a3 = np.sqrt(2)*np.array([0,1/2,1/2])

        i,j,k = np.mgrid[-2*Rrel:2*Rrel,-2*Rrel:2*Rrel,-2*Rrel:2*Rrel]

        testpts = 2*self.a*(np.outer(i.flatten(),a1) + np.outer(j.flatten(),a2) + np.outer(k.flatten(),a3))
        
        #finds the radius at each of those points
        rs = LA.norm(testpts, axis=1)
        
        #checks if each bead is close to the radius
        isRad = (np.absolute(rs - Rrel*2*a) < 0.75*self.a)
        
        #only picks out beads near the radius
        self.pts = testpts[isRad]
        self.N = np.size(self.pts,0)
    
    def copy(self):
        return Sphere(self.Rrel,a=self.a)
    
    #other useful quantities
    
    def radius(self):
        rs = LA.norm(self.pts, axis=1)
        return np.mean(rs) + Rcorr*self.a
    
    def surfaceDensity(self):
        return self.N/(np.pi*(self.radius**2))

#generates a cylinder of beads
class Cylinder(Collection):
    
    def __init__(self,s,Nr,a=1,mode='A'):
        self.a = a
        self.s = s
        self.Nr = Nr
        self.mode = mode
        self.N = s*Nr
        
        #creates an index of points for easy vectorization
        nums = np.arange(Nr*s)
        thetas = np.copy(nums)
        rs = np.copy(nums)
        zs = np.copy(nums)
        
        #stacking mode refers to whether the beads in a layer are colinear (A) or staggered (B)
        
        if(mode=='A'):
            #there are s beads in a row, so the thetas need to be 2pi/s apart
            thetas = np.floor(nums/Nr)*(2*np.pi/s)
            #the rs are a constant array at distance a/sin(pi/s)
            rs = np.ones(Nr*s)*(a/np.sin(np.pi/s))
            #the zs increase by 2a each row
            zs = (nums%Nr)*(2*self.a)
        
        if(mode == 'B'):
            #when the beads are staggered every other row is rotated forward pi/s
            thetas = np.floor(nums/Nr)*(2*np.pi/s) + nums%2*(np.pi/s)*(np.pi/s)
            #rs are the same
            rs = np.ones(Nr*s)*(a/np.sin(np.pi/s))
            #zs are slightly closer together, rt(3)a instead of 2a apart
            zs = (nums%Nr)*(np.sqrt(3)*self.a)
        
        #converts to cartesian coordinates
        cyls = np.transpose(np.array([thetas,rs,zs]))
        cars = np.array(list(map(toCar,cyls))).round(8)
        
        self.pts = cars
        
        #centers cylinder on the origin
        if (mode == 'A'):
            self.translate(np.array([0,0,-2*self.a*(Nr-1)/2]))
        if (mode == 'B'):
            self.translate(np.array([0,0,-np.sqrt(3)*self.a*(Nr-1)/2]))
            
    def copy(self):
        return Cylinder(self.s,self.Nr,a=self.a,mode=self.mode)
    
    #these are all standard calculations you can do on a cylindrical shell of beads
    
    def radius(self):
        return self.a/np.sin(np.pi/self.s) + self.a
    
    def length(self):
        if(self.mode == 'A'):
            return self.Nr*2*self.a
        if(self.mode == 'B'):
            return self.Nr*np.sqrt(3)*self.a + 2*self.a
    
    def aspectRatio(self):
        return self.length()/(2*self.radius())
    
    def surfaceDensity(self):
        return self.s*self.Nr/(2*np.pi*self.radius()*self.length())
    

class Disc(Collection):
    
    def __init__(self,s,Nr,a=1,mode='A'):
        
        if(Nr>s/2):
            raise Exception("Please use the Cylinder Class for speedier calculations")
        
        self.a = a
        self.s = s
        self.Nr = Nr
        self.mode = mode
        self.N = s*Nr
        
        #creates an index of points for easy vectorization
        nums = np.arange(Nr*s)
        thetas = np.copy(nums)
        rs = np.copy(nums)
        zs = np.copy(nums)
        
        #stacking mode refers to whether the beads in a layer are colinear (A) or staggered (B)
        
        if(mode=='A'):
            #there are s beads in a row, so the thetas need to be 2pi/s apart
            thetas = np.floor(nums/Nr)*(2*np.pi/s)
            #the rs are a constant array at distance a/sin(pi/s)
            rs = np.ones(Nr*s)*(a/np.sin(np.pi/s))
            #the zs increase by 2a each row
            zs = (nums%Nr)*(2*self.a)
        
        if(mode == 'B'):
            #when the beads are staggered every other row is rotated forward pi/s
            thetas = np.floor(nums/Nr)*(2*np.pi/s) + nums%2*(np.pi/s)*(np.pi/s)
            #rs are the same
            rs = np.ones(Nr*s)*(a/np.sin(np.pi/s))
            #zs are slightly closer together, rt(3)a instead of 2a apart
            zs = (nums%Nr)*(np.sqrt(3)*self.a)
        
        #converts to cartesian coordinates
        cyls = np.transpose(np.array([thetas,rs,zs]))
        cars = np.array(list(map(toCar,cyls))).round(8)
        
        self.pts = cars
        
        #centers cylinder on the origin
        if (mode == 'A'):
            self.translate(np.array([0,0,-2*self.a*(Nr-1)/2]))
        if (mode == 'B'):
            self.translate(np.array([0,0,-np.sqrt(3)*self.a*(Nr-1)/2]))
        
        #makes hcp caps
        Rrel = np.floor((cyls[0,1]-a)/a)
        a1 = np.array([1/2,np.sqrt(3)/2,0])
        a2 = np.array([1/2,-np.sqrt(3)/2,0])
        
        i,j = np.mgrid[-2*Rrel:2*Rrel,-2*Rrel:2*Rrel]

        testpts = 2*self.a*(np.outer(i.flatten(),a1) + np.outer(j.flatten(),a2))
        
        #finds the radius at each of those points
        rs2 = LA.norm(testpts, axis=1)
        
        #checks if each bead is within a maximum radius Rrel
        inside = (np.absolute(rs2) < (Rrel+0.25)*self.a)
        
        #only picks out beads within said radius
        cap = testpts[inside]
        
        # adds them to the collection at the ends
        z = np.max(self.pts[:,2])
        self.add(cap+np.array([0,0,z]))
        self.add(cap+np.array([0,0,-z]))
        
    def copy(self):
        return Disc(self.s,self.Nr,a=self.a,mode=self.mode)
    
    #these are all standard calculations you can do on a cylindrical shell of beads
    
    def radius(self):
        return self.a/np.sin(np.pi/self.s) + self.a
    
    def width(self):
        if(self.mode == 'A'):
            return self.Nr*2*self.a
        if(self.mode == 'B'):
            return self.Nr*np.sqrt(3)*self.a + 2*self.a
    
    def aspectRatio(self):
        return (2*self.radius())/self.width() 
    

#generates a helix of beads
class Helix(Collection):
    
    def __init__(self,s,p,Np,a=1,mode='R'):
        self.a = a
        self.s = s
        self.p = p # here p is pitch, not aspect ratio
        self.Np = Np # number of pitches, may be a fraction
        self.N = np.round(s*Np)
        self.mode = mode
        
        #creates an index of points for easy vectorization
        nums = np.arange(self.N)
        thetas = np.copy(nums)
        rs = np.copy(nums)
        zs = np.copy(nums)
        
        if (mode == 'R'):
            chi = 1
        elif (mode == 'L'):
            chi = -1
        else:
            raise Exception("Please specifiy valid chirality; \'R\' or \'L\'")
            
        #there are s beads in a row, so the thetas need to be 2pi/s apart
        thetas = (nums%s)*(2*np.pi/s)
        #the rs are a constant array at distance a/sin(pi/s)
        rs = np.ones(self.N)*(a/np.sin(np.pi/s))
        #the zs increase by 2a each row
        zs = chi*nums*p/s
        
        #converts to cartesian coordinates
        cyls = np.transpose(np.array([thetas,rs,zs]))
        cars = np.array(list(map(toCar,cyls))).round(8)
        
        self.pts = cars
        
        #centers helix on the origin
        self.translate(np.array([0,0,-chi*self.N/2*p/s]))
            
    def copy(self):
        return Helix(self.s,self.p,self.Np,a=self.a,mode=self.mode)
    
    #these are all standard calculations you can do on a cylindrical shell of beads
    
    def radius(self):
        return self.a/np.sin(np.pi/self.s) + self.a
    
    def length(self):
        return np.abs((self.pts[-1,2]-pts[0,2])) + self.a
    
    def aspectRatio(self):
        return self.length()/(2*self.radius())
    
# easy code to merge two collections, simply make a new one with the two lists of point joined
def merge(col1,col2):
    if(col1.a == col2.a):
        pts1 = np.copy(col1.pts)
        pts2 = np.copy(col2.pts)
        pts3 = np.vstack((pts1,pts2))
        return Collection(pts3,a=col1.a)
    else:
        print("different particle sizes, cannot merge")
        return None