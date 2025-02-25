import h5py
f = h5py.File("C:\\Users\\amjad\\Downloads\\data\\data\\ecg_tracings.hdf5",'r')
for item in f:
    print (item) #+ ":", f[item]