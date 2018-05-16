import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
def extract_log(log_file,new_log_file,key_word):
    f = open(log_file)
    train_log = open(new_log_file, 'w')
    for line in f:
        # 去除多gpu的同步log
        if 'Syncing' in line:
            continue
        # 去除除零错误的log
        if 'nan' in line:
            continue
        if key_word in line:
            train_log.write(line)

    f.close()
    train_log.close()

extract_log('log/coco_voc_kitti_train_14.log','avg_loss.log','images')

lines =1878760
result = pd.read_csv('avg_loss.log',error_bad_lines=False, names=['loss', 'avg', 'rate', 'seconds', 'images'])
result.head()

result['loss']=result['loss'].str.split(' ').str.get(1)
result['avg']=result['avg'].str.split(' ').str.get(1)
result['rate']=result['rate'].str.split(' ').str.get(1)
result['seconds']=result['seconds'].str.split(' ').str.get(1)
result['images']=result['images'].str.split(' ').str.get(1)
result.head()
result.tail()

#print(result['loss'])
#print(result['avg'])
#print(result['rate'])
#print(result['seconds'])
#print(result['images'])

result['loss']=pd.to_numeric(result['loss'])
result['avg']=pd.to_numeric(result['avg'])
result['rate']=pd.to_numeric(result['rate'])
result['seconds']=pd.to_numeric(result['seconds'])
result['images']=pd.to_numeric(result['images'])
result.dtypes

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(result['avg'].values,label='avg_loss')
ax.legend(loc='best')
ax.set_title('The loss curves')
ax.set_xlabel('batches')
fig.savefig('avg_loss')
