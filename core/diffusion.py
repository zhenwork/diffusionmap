import copy
import numpy as np
from core.dataobject import MeasureBase,IdMap
from core.neighbors import Neighbors_Laplacian 

# TODO: unify the definition of algorithm-function mapping
# TODO: check symmetry algorithm too naive
# TODO: checkNone() function too complex
# TODO: data cluster and Ds matrix co-existing problem, simply providing Ds matrix doesn't support map.__getitem__() search.

class DMFeature:
    def __init__(self):
        pass
    def parabola_polar(self, points=None, eigx=1, eigy=2):
        """
        points: [(x1,y1),(x2,y2),...,(xn,yn)]
        """
        if points is None:
            points = [(self.eigenvalue[eigx]*self.eigenvector[idx, eigx], \
                       self.eigenvalue[eigy]*self.eigenvector[idx, eigy] ) \
                       for idx in range(len(self.eigenvalue))]

        N = len(points)
        xlist = []
        ylist = []
        for i in range(N):
            xlist.append(points[i][0])
            ylist.append(points[i][1])
        cx = np.mean(np.array(xlist))
        cy = np.mean(np.array(ylist))
        
        p = []
        for i in range(N):
            p.append( points[i][0]+1.j*points[i][1] - (cx+1.j*cy) )
        
        angle = np.angle(p)*180./np.pi + 180
        
        theta = np.arange(360)
        var = []
        for i in theta:
            Tangle = (angle+i)%360
            var.append(np.amax(Tangle) - np.amin(Tangle))   #(np.std(Tangle))
        var = np.array(var)
        index = np.argmin(var)
        Nangle = (angle + theta[index])%360 
        
        return Nangle-np.amin(Nangle)



class DiffusionMap(DMFeature):
    """
    ! ONLY applicable to small datasets

    matrix Ds: the root square of difference (or L2 norm). You need to perform Ds^2 sometimes, refer to kernel function
    matrix Lp: the similarity matrix
    alpha: meaure the type of diffusion
    sigma: scaling of Distance matrix (Ds)
    nearest: keep the nearest neighbors
    """
    def __init__(self, alpha=0.5, Lp=None, Ds=None, sigma=None, nearest=None, nearest_alg="fair", sigma_alg="chuck", kernel_name="gaussian", \
                       measure=MeasureBase("l2")):
        
        self.Ds = Ds
        self.Lp = Lp
        self.cluster = []       ## {key:DataBase}
        self.map = IdMap()
        #self.notes = {}        ## {"knownlabel":}
        self.alpha = alpha
        self.sigma = sigma
        self.nearest = nearest
        self.measure = measure
        self.sigma_alg = sigma_alg
        self.nearest_alg = nearest_alg
        self.kernel_name = kernel_name
        
    def addpoint(self, data=None, stamp=None):
        """
        default idx is 0,1,2,3
        """
        if stamp is not None:
            data.stamp = stamp
        self.cluster.append(data)
        self.map.memorize(data)
        return True
    
    def checkSymmetry(self, mat):
        if mat.shape != mat.T.shape:
            return False
        if np.sum(np.abs(mat-mat.T))>1e-4:
            return False
        return True

    def symmetrize(self, mat):
        """
        This function only deals with cases when half of matrix is filled in (including the diagonal).
        """
        if not self.checkSymmetry(mat):
            return mat + mat.T - np.diag(mat.diagonal())
        else:
            return mat

    def keepNeighbors(self, Lp=None, nearest=None, nearest_alg=None):
        ## ONLY work for the Laplacian matrix, which is similarity measure
        ## Nearest means the similarity, or the largest values in matrix
        sym_Lp = self.symmetrize(Lp)
        NLP = Neighbors_Laplacian(Lp=sym_Lp, nearest=nearest, algorithm=nearest_alg)
        return NLP.prune()

    def normalize(self, arr):
        mean = np.mean(arr)
        std = np.std(arr)
        return (arr - mean) / std

    def kernel(self, kernel_name=None, **kwargs):
        kernel_name = self.checkNone(kernel_name=kernel_name)
        Ds = kwargs["Ds"]
        sigma = kwargs["sigma"]

        if kernel_name.lower() == "gaussian":
            return np.exp( - 1.0*Ds**2 / sigma)
        else:
            return None

    def checkNone_backup(self, **kwargs):
        returns = []
        for item in kwargs:
            if (kwargs[item] is None) and hasattr(self, item):
                returns.append(getattr(self, item))
            else:
                returns.append(kwargs[item])
        return returns 

    def checkNone(self, **kwargs):
        key = kwargs.keys()[0]
        if kwargs[key] is None and hasattr(self, key):
            return getattr(self, key)
        else:
            return kwargs[key]
        
    def LaplacianMapping(self, Lp=None, alpha=None, nearest=None, nearest_alg=None):
        """
        Lp must be similarity matrix, symmetrized
        """
        Lp = self.checkNone(Lp=Lp)
        alpha = self.checkNone(alpha=alpha) 
        nearest = self.checkNone(nearest=nearest)
        nearest_alg = self.checkNone(nearest_alg=nearest_alg) 

        _Lp = Lp.copy()
        _Lp = self.keepNeighbors(Lp=_Lp, nearest=nearest, nearest_alg=nearest_alg)
        
        N = len(_Lp)
        D12 = np.zeros((N,N))
        D121 = np.zeros((N,N))

        for i in range(N):
            D12[i,i] = np.sum(_Lp[i,:])**(-alpha)
            
        L12 = np.dot(D12, np.dot(_Lp, D12))

        for i in range(N):
            D121[i,i] = np.sum(L12[i,:])**(-1.)
        M = np.dot(D121, L12)
        return M


    def runDiffusionMap(self, Lp=None, alpha=None, Ds=None, sigma=None, nearest=None, nearest_alg=None, sigma_alg=None, kernel_name=None):
        
        Lp = self.checkNone(Lp=Lp)
        alpha = self.checkNone(alpha=alpha)
        Ds = self.checkNone(Ds=Ds)
        sigma = self.checkNone(sigma=sigma) 
        nearest = self.checkNone(nearest=nearest)
        nearest_alg = self.checkNone(nearest_alg=nearest_alg)
        sigma_alg = self.checkNone(sigma_alg=sigma_alg)
        kernel_name = self.checkNone(kernel_name=kernel_name)

        if Lp is None and Ds is None:
            return None
        if Lp is not None:
            ## has Lp matrix
            M = self.LaplacianMapping(Lp=Lp, alpha=alpha, nearest=nearest, nearest_alg=nearest_alg)
            val, vec = np.linalg.eig(M)
            argval = np.argsort(val)[::-1]
            self.eigenvector = np.real(vec[:,argval])
            self.eigenvalue = np.real(val[argval])
            return self.eigenvalue, self.eigenvector

        if sigma is None:
            ## has Ds matrix, check sigma
            sigma = getattr(self, "sigma_alg_"+str(sigma_alg))(kernel_name=kernel_name, Ds=Ds)
        
        self.Lp = self.kernel(kernel_name=kernel_name, Ds=Ds, sigma=sigma)
        M = self.LaplacianMapping(Lp=self.Lp, alpha=alpha, nearest=nearest, nearest_alg=nearest_alg)

        val, vec = np.linalg.eig(M)
        argval = np.argsort(val)[::-1]
        self.eigenvector = np.real(vec[:,argval])
        self.eigenvalue = np.real(val[argval])
        return self.eigenvalue, self.eigenvector
        

    def visualize(self, eigx=1, eigy=2, eigenvector=None, eigenvalue=None, label=None, figsize=(8,8)):
        eigenvector = self.checkNone(eigenvector=eigenvector)
        eigenvalue = self.checkNone(eigenvalue=eigenvalue)

        ## max and min of scatters in the plot
        (minx, maxx) = ( np.amin(eigenvalue[eigx]*eigenvector[:,eigx]), np.amax(eigenvalue[eigx]*eigenvector[:,eigx]) )
        (miny, maxy) = ( np.amin(eigenvalue[eigy]*eigenvector[:,eigy]), np.amax(eigenvalue[eigy]*eigenvector[:,eigy]) )
        gapx = maxx - minx
        gapy = maxy - miny

        ## px,py saves the position of x,y axis
        px = eigenvalue[eigx]*eigenvector[:,eigx].ravel()
        py = eigenvalue[eigy]*eigenvector[:,eigy].ravel()

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(px, py)

        if label is not None:
            for i, txt in enumerate(label):
                ax.annotate(txt, (px[i], py[i]))

        #plt.plot(px, py, '.', ms=8, color='b')
        plt.xlim(minx-gapx*0.1, maxx+gapx*0.1)
        plt.ylim(miny-gapy*0.1, maxy+gapy*0.1)
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        plt.xlabel("Dimension " + str(eigx))
        plt.ylabel("Dimension " + str(eigy))
        plt.tight_layout()
        plt.show()


    def measureMatrix(self, measure=None, verbose=True):
        measure = self.checkNone(measure=measure)

        N = len(self.cluster)
        done = 0
        total_jobs = N*(N-1)/2 + N
        print_points = np.linspace(0,total_jobs,11).astype(int)
        
        N = len(self.cluster)
        meaMatrix = np.zeros((N,N))
        for i, item1 in enumerate(self.cluster): 
            self.map.memorize(item1,idx=i)
            for j, item2 in enumerate(self.cluster):
                if i < j:
                    continue
                d = measure.measure()(item1, item2)
                meaMatrix[i,j] = d
                meaMatrix[j,i] = d
                done += 1
                if verbose and done in print_points:
                    print "## Finish %3d/100"%(round(100.*done/total_jobs))
        return meaMatrix
    

    def insertMatrix(self, data=None, to="Ds", meaure=None):
        """
        Testing stage, it may not work
        """
        self.insert = DiffusionMap(alpha=self.alpha, Lp=self.Lp, Ds=self.Ds, sigma=self.sigma, \
                        nearest=self.nearest, nearest_alg=self.nearest_alg, sigma_alg=self.sigma_alg, \
                        kernel_name=self.kernel_name, measure=self.measure)
        self.insert.map = copy.deepcopy(self.map)
        
        for d in data: 
            self.insert.addpoint(d)
        if meaure is None:
            measure = self.insert.measure

        n = len(self.insert.cluster)
        N = len(self.cluster)
        newMat = np.zeros((N+n, N+n))
        newMat[:N,:N] = getattr(self, to).copy()
        for ii,item1 in enumerate(self.insert.cluster):
            i = ii+N 
            self.insert.map.memorize(item1,idx=i)
            for j,item2 in enumerate(self.cluster):
                d = measure.measure()(item1, item2)
                newMat[i, j] = d
                newMat[j, i] = d
        return newMat

        

    def estimateMap(self, data, measure=None):
        """
        Estimate the vector of a new point without running it again
        """
        if measure is not None:
            measure = self.measure
        return
    

    def sigma_alg_chuck(self, Ds=None, kernel_name=None, verbose=False):
        Ds = self.checkNone(Ds=Ds)
        kernel_name = self.checkNone(kernel_name=kernel_name)

        logEps = np.linspace(-20.0, 20.0, num=80)
        logSampleDistance = logEps[1]-logEps[0]
        eps = np.exp(logEps)
        L = np.zeros_like(logEps)
        for i,e in enumerate(eps):
            K = self.kernel(kernel_name=kernel_name, Ds=Ds, sigma=e)
            L[i] = np.log(np.sum(K)+1e-10)

        if verbose:
            import matplotlib.pyplot as plt
            plt.plot(logEps, L)
            plt.tight_layout()
            plt.show()

        normL = L - np.min(L) 
        normL /= np.max(normL) 
        gradients=np.gradient(normL)
        maxInd=np.argmax(gradients)
        
        if maxInd < len(gradients)-1:
            gradient = (normL[maxInd+1]-normL[maxInd]) / logSampleDistance
            targetY = 0.75
            sigmaK = np.exp((targetY - normL[maxInd]) / gradient  + logEps[maxInd])
        print "sigma=%f"%sigmaK
        return sigmaK