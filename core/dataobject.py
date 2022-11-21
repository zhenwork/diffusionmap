import numpy as np 
from core.utils import combine_dict_to_list

class DataBase:
    """
    data: real data such as numpy array.
    fname: can be a file of saved data, if activated, you need to define the self.Data() function loading your data file
    stamp: give a description of this data point, such as timestamp 
    """
    nextid = 0
    def __init__(self, data=None, fname=None, stamp=None):
        self.data = data
        self.fname = fname
        self.id = DataBase.nextid
        self.stamp = stamp 
        DataBase.nextid += 1 
    def Data(self):
        if self.data is None:
            return self.load()
        return self.data
    def load(self):
        ## user can define different load functions
        self.data = np.load(self.fname)
        return self.data
    def clear(self):
        self.__init__()
    def show(self):
        print self.__dict__
    def define(self, function, method="load"):
        setattr(self, str(method), function)
        
class IdMap(object):
    """
    # ONLY keep id (unique id), stamp (user defined label), idx (position in diffusion matrix), fname (file name)
    # TODO: addmap() is too complex
    """
    def __init__(self, newkeys=None):
        self.support = ["id", "idx", "stamp", "fname"]
        self.datapoints = {}  ## id:{id:, idx:, stamp:, fname:}
        self.addkey(newkeys)
        # for i in self.support:
        #     for j in self.support:
        #         if i==j: continue
        #         setattr(self, i+"_to_"+j, {}) 

    def __len__(self):
        return len(self.datapoints)
        
    def addkey(self, keys=None):
        if keys is None:
            return
        if isinstance(keys, str):
            if keys not in self.support:
                self.support.append(keys)
            return
        for each in keys:
            if each not in self.support:
                self.support.append(each)
        return 

    def memorize(self, datapoint=None, **kwargs):
        if datapoint is not None:
            db = self.capture(datapoint)
            db.update(kwargs)
            self.datapoints[db["id"]] = db
            return
        else:
            if len(kwargs)==0 or "id" not in kwargs:
                return
            elif kwargs["id"] in self.datapoints:
                self.datapoints[kwargs["id"]].update(kwargs)
            else:
                self.datapoints[kwargs["id"]] = kwargs

    def capture(self, datapoint):
        dictionary = dict.fromkeys(self.support)
        for key in dictionary:
            if hasattr(datapoint, key):
                dictionary[key] = getattr(datapoint, key)
        return dictionary

    def __getitem__(self, **kwargs):
        combine = []
        for db_id in sorted(self.datapoints):
            datapoint = self.datapoints[db_id]
            exists = True
            for key in kwargs:
                if key not in datapoint and kwargs[key] is not None:
                    exists=False
                    break
                if key in datapoint and datapoint[key]!=kwargs[key]:
                    exists=False
                    break
            if exists:
                combine.append(datapoint)
        if len(combine)==0:
            return None
        return combine_dict_to_list(combine)


class DataCluster(IdMap):
    def __init__(self):
        super(DataCluster,self).__init__()
        self.cluster = []
    def addpoint(self, data=None, stamp=None):
        """
        default idx is 0,1,2,3
        """
        if stamp is not None:
            data.stamp = stamp
        self.cluster.append(data)
        self.memorize(data,idx=len(self.cluster)-1)
        return True


    # def memorize(self, datapoint=None, **kwargs):
    #     if data is None:
    #         return
    #     elif isinstance(data, dict):
    #         self.addmap_dict(**data)
    #         return
    #     elif isinstance(data, class):
    #         self.addmap_data(data)
    #         return

    # def addmap_data(self, data):
    #     for i in self.support:
    #         for j in self.support:
    #             if (i==j) or (not hasattr(data, i)) or (not hasattr(data, j)): 
    #                 continue
    #             if getattr(data, i) in getattr(self, i+"_to_"+j):
    #                 if isinstance(getattr(self, i+"_to_"+j)[getattr(data, i)], list):
    #                     getattr(self, i+"_to_"+j)[getattr(data, i)].append(getattr(data, j))
    #                 else:
    #                     getattr(self, i+"_to_"+j)[getattr(data, i)] = list([getattr(self, i+"_to_"+j)[getattr(data, i)]])
    #                     getattr(self, i+"_to_"+j)[getattr(data, i)].append(getattr(data, j))
    #             else:
    #                 getattr(self, i+"_to_"+j)[getattr(data, i)] = getattr(data, j)
    # def addmap_dict(self, **kwargs):
    #     """
    #     kwargs: {"idx":10, "feature1":20, "feature2":30} or {"idx":[10,20], "feature1":[20,20], "feature2":[5,5]}
    #     """ 
    #     for i in kwargs.keys():
    #         for j in kwargs.keys():
    #             if (i==j):
    #                 continue
    #             if not hasattr(self, i+"_to_"+j):
    #                 setattr(self, i+"_to_"+j, {}) 
    #             if kwargs[i] in getattr(self, i+"_to_"+j):
    #                 if isinstance(getattr(self, i+"_to_"+j)[kwargs[i]], list):
    #                     getattr(self, i+"_to_"+j)[kwargs[i]].append(kwargs[j])
    #                 else:
    #                     getattr(self, i+"_to_"+j)[kwargs[i]] = list([getattr(self, i+"_to_"+j)[kwargs[i]]])
    #                     getattr(self, i+"_to_"+j)[kwargs[i]].append(kwargs[j])
    #             else:
    #                 getattr(self, i+"_to_"+j)[kwargs[i]] = kwargs[j]

 
class MeasureBase:
    def __init__(self, method=None,**kwargs):
        self.method = method
        self.support = ["l1","l2","cosine"]
        self.kwargs = kwargs

    def abbrev(self, method):
        if method.lower() in ["l2","euclidean"]:
            return "euclidean"
        elif method.lower() in ["l1"]:
            return "l1"
        else:
            return method
    def measure(self, method=None):
        if method is None:
            method = self.method
        method = self.abbrev(method)
        return getattr(self, method)

    def rms(self, data1, data2, vmin=-1e9, vmax=1e9, norm=True):
        if "vmin" in self.kwargs: vmin = self.kwargs["vmin"]
        if "vmax" in self.kwargs: vmax = self.kwargs["vmax"]
        if "norm" in self.kwargs: norm = self.kwargs["norm"]

        x1 = np.array(data1.Data())
        x2 = np.array(data2.Data())

        index = np.where((x1>=vmin)&(x1<vmax)&(x2>=vmin)&(x2<vmax))

        x1 = x1[index].copy()
        x2 = x2[index].copy()

        if norm:
            x1 = (x1-np.mean(x1))/np.std(x1)
            x2 = (x2-np.mean(x2))/np.std(x2)

        return np.sqrt(np.sum((x1 - x2)**2)*1.0/len(x1))

    def normL2(self, data1, data2):
        x1 = np.array(data1.Data())
        x2 = np.array(data2.Data())

        x1 = (x1-np.mean(x1))/np.std(x1)
        x2 = (x2-np.mean(x2))/np.std(x2)
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def euclidean(self, data1, data2, vmin=-1e9, vmax=1e9, norm=False):
        if "vmin" in self.kwargs: vmin = self.kwargs["vmin"]
        if "vmax" in self.kwargs: vmax = self.kwargs["vmax"]
        if "norm" in self.kwargs: norm = self.kwargs["norm"]

        x1 = np.array(data1.Data())
        x2 = np.array(data2.Data())

        index = np.where((x1>=vmin)&(x1<vmax)&(x2>=vmin)&(x2<vmax))

        x1 = x1[index].copy()
        x2 = x2[index].copy()

        if norm:
            x1 = (x1-np.mean(x1))/np.std(x1)
            x2 = (x2-np.mean(x2))/np.std(x2)

        return np.sqrt(np.sum((x1 - x2)**2))

    def define(self, function, method="user"):
        setattr(self, str(method), function)
        self.support.append(method)