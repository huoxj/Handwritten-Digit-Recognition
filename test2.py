import numpy as np
import packages.AutoDiffrentiation as ad
import numericaltest as ntest
import time
import pickle

class obj:
    dic = {"ww":1, "kk":2}
    name = "wowmyname"
    def experess(self):
        print(self.dic)
        print(self.name)

mono = obj()
s = pickle.dumps(mono)
with open('temp.pickle','wb') as f:
    pickle.dump(s, f)
with open('temp.pickle','rb') as f:
    btys = pickle.load(f)
    data = pickle.loads(btys)
print(type(data))
#test