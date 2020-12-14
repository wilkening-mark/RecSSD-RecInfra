import numpy as np
import matplotlib.pyplot as plt

#for k in range(4):
k=1

scale = 1.0
nsamples = 1000
table_size = 1000000
cacheline_size = 128

s = np.random.exponential(scale, nsamples)

nbins = 50
count, bins = np.histogram(s, nbins, density=True)
nsamples = np.sum(count)

count[0] += int(nsamples*k)

nsamples = np.sum(count)

list_sd = []
cumm_sd = []
cumm_count = 0.
for i in xrange(len(count)):
    cumm_count += count[i]
    cumm = float(cumm_count) / nsamples
    list_sd.append(i)
    cumm_sd.append(cumm)
cumm_sd[-1] = 1.0

list_sd_str = ""
for sd in list_sd[:-1]:
    list_sd_str += str(sd) + ", "
list_sd_str += str(list_sd[-1]) + "\n"
cumm_sd_str = ""
for cumm in cumm_sd[:-1]:
    cumm_sd_str += str(cumm) + ", "
cumm_sd_str += str(cumm_sd[-1]) + "\n"

access_list = [i for i in xrange(table_size//cacheline_size)]
access_str = ""
for access in access_list[:-1]:
    access_str += str(access) + ", "
access_str += str(access_list[-1]) + "\n"

#dist_file = open("dist_"+str(k)+".log", "w")
dist_file = open("dist.log", "w")
dist_file.write(access_str+list_sd_str+cumm_sd_str)
dist_file.close()
