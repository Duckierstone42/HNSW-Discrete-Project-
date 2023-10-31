import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
import scipy.optimize as opt

with open("datacosine.pickle","rb") as fp:
    cosine_data = pickle.load(fp)
    cosine_x = np.array(cosine_data[0])
    cosine_y = np.array(cosine_data[1])

with open("dataip.pickle","rb") as fp:
    ip_data = pickle.load(fp)
    ip_x = np.array(ip_data[0])
    ip_y = np.array(ip_data[1])
with open("datal2.pickle","rb") as fp:
    l2_data = pickle.load(fp)
    l2_x = np.array(l2_data[0])
    l2_y = np.array(l2_data[1])
generic_fig = plt.figure()


#cosine_y_smoothed = gaussian_filter1d(cosine_y,sigma=4)
#ip_y_smoothed = gaussian_filter1d(ip_y,sigma=4)
#l2_y_smoothed = gaussian_filter1d(l2_y,sigma=4)

#plt.plot(cosine_x,cosine_y_smoothed,color="r",label="Cosine")
#plt.plot(ip_x,ip_y_smoothed,color="g",label="IP")
#plt.plot(l2_x,l2_y_smoothed,color="b",label="L2")

#Logistic regression model for cosine
def f(x, a, b):
    
    return a * np.log2(x) + b
popt1, _ = opt.curve_fit(f, cosine_x, cosine_y)
y_fit_cosine = f(cosine_x, *popt1)
print("Cosine:",popt1[0],popt1[1])
popt2,_ = opt.curve_fit(f,ip_x,ip_y)
y_fit_ip = f(ip_x,*popt2)
popt3,_ = opt.curve_fit(f,l2_x,l2_y)
y_fit_l2 = f(l2_x,*popt3)
print("IP:",popt2[0],popt2[1])
print("L2:",popt3[0],popt3[1])

#plt.plot(ip_x,ip_y, color="g",label="IP")
#plt.plot(ip_x,y_fit_ip, color="black",label="IP Logarithmic Regression")
plt.plot(l2_x,l2_y, color="b",label="L2")
plt.plot(l2_x,y_fit_l2, color="black",label="L2 Logarithmic Regression")
#plt.plot(cosine_x,cosine_y, color="r",label="Cosine")
#plt.plot(cosine_x,y_fit_cosine, color="black",label="Cosine Logarithmic Regression")


plt.xlabel("Number of elements in HNSW")
plt.ylabel("Time in milliseconds")
plt.legend(loc = "upper left")
plt.title(f"Time Required to search up 100 randomly chosen elements \n in HNSW graphs of varying sizes, \n using the L2 distance metric")
plt.show()
