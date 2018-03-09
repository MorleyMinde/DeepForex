import numpy as np
import h5py, os, time

def getdatasets(key,archive):

  if key[-1] != '/': key += '/'

  out = []

  for name in archive[key]:

    path = key + name

    if isinstance(archive[path], h5py.Dataset):
      out += [path]
    else:
      try   : out += getdatasets(path,archive)
      except: pass

  return out

data  = h5py.File('training/training3/weights.h5' ,'r')
fixed = h5py.File('training/training3/weights2.h5','w')

datasets = getdatasets('/',data)

groups = list(set([i[::-1].split('/',1)[1][::-1] for i in datasets]))
groups = [i for i in groups if len(i)>0]

idx    = np.argsort(np.array([len(i.split('/')) for i in groups]))
groups = [groups[i] for i in idx]

for group in groups:
  fixed.create_group(group)

for path in datasets:

  # - check path
  if path not in data: continue

  # - try reading
  try   : data[path]
  except: continue

  # - get group name
  group = path[::-1].split('/',1)[1][::-1]

  # - minimum group name
  if len(group) == 0: group = '/'

  # - copy data
  data.copy(path, fixed[group])