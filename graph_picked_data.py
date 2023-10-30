import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

with open("datacosine.pickle","rb") as fp:
    cosine_data = pickle.load(fp)
    cosine_x = cosine_data[0]
    cosine_y = cosine_data[1]

with open("dataip.pickle","rb") as fp:
    ip_data = pickle.load(fp)
    ip_x = ip_data[0]
    ip_y = ip_data[1]
with open("datal2.pickle","rb") as fp:
    l2_data = pickle.load(fp)
    l2_x = l2_data[0]
    l2_y = l2_data[1]
generic_fig = plt.figure()


cosine_y_smoothed = gaussian_filter1d(cosine_y,sigma=4)
ip_y_smoothed = gaussian_filter1d(ip_y,sigma=4)
l2_y_smoothed = gaussian_filter1d(l2_y,sigma=4)

plt.plot(cosine_x,cosine_y_smoothed,color="r",label="Cosine")
plt.plot(ip_x,ip_y_smoothed,color="g",label="IP")
plt.plot(l2_x,l2_y_smoothed,color="b",label="L2")

#plt.plot(cosine_x,cosine_y, color="r",label="Cosine")
#plt.plot(ip_x,ip_y, color="g",label="IP")
#plt.plot(l2_x,l2_y, color="b",label="L2")
plt.xlabel("Number of elements in HNSW")
plt.ylabel("Time in milliseconds")
plt.legend(loc = "upper right")
plt.title(f"Time Required to search up 100 randomly chosen elements \n in HNSW graphs of varying sizes, \n calculated via various distance metrics")
plt.show()
