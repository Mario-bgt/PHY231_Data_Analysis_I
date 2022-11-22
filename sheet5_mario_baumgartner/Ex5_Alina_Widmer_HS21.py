import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy import optimize


cur = np.loadtxt("current_measurements.txt")
cur_unc = np.loadtxt("current_measurements_uncertainties.txt")

v = cur[:,0]
a = cur[:,1]
unc_a = cur_unc[:,2]

#1a)
plt.scatter(v, a)
plt.grid()
plt.ylabel("Current [A]")
plt.xlabel("Voltage [V]")
plt.title("current vs. voltage measurements")
plt.savefig("1a_current_voltage.png")
#plt.show()
plt.clf()

#1b)
def chi(r):
    X2 = 0
    for i in range(0,5):
        X2 = X2 + (a[i]-v[i]/r)**2
    return X2

#1c)
x = np.linspace(1.5, 2, 500)
chi_list = []
for i in range(0,500):
    n = chi(x[i])
    chi_list.append(n)


plt.plot(x, chi_list)
plt.grid()
plt.xlabel("Resistance [Ohm]")
plt.ylabel("Chi squared")
plt.title("Chi squared of resistance")
plt.savefig("1c_chi_squared_resistance.png")
#plt.show()
plt.clf()

min_chi = min(chi_list)
best_fit = []
for i in range(0, 500):
    if chi(x[i]) == min_chi:
        best_fit.append(x[i])


print("1c) The minimal Chi squared value is at "+ str(round(min(chi_list), 2)) +
      ". This can also be read from the plot directly (look at extremum, minimum). "
      "The best fit value for the resistance is at 1.73 Ohm, which can be read from the plot.")

#1d)
#for C = 0
mean_v = np.mean(v)
mean_a = np.mean(a)
mean_va = np.mean(v*a)
m = 1/((mean_va-(mean_v*mean_a))/(np.mean(v**2)-mean_v**2))
print("1d) The value I obtain for the analytical formula reads " +str(round(m, 2)) +" Ohm, which does not agree with the value 1.73 Ohm from 1c)."
                                                                               "Therefore, the fit is not very good.")
#1e)
uncert_chi = min(chi_list)+1
print("1e) At the chi squared value of 1.78 (min chi^2 + 1) the corresponding R-value reads ~1.53 Ohm. Therefore, 1.73 Ohm - 1.53 Ohm = 0.2 Ohm (= our uncertainty). "
      "However, does not yet agree with formula shown in lecture and 1.73+- 0.20 Ohmis not within the range of 2.05 Ohm.")

#2a)
def chi_un(r):
    X2 = 0
    for i in range(0,5):
        X2 = X2 + ((a[i]-v[i]/r)**2)/(unc_a[i]**2)
    return X2

#2b)

x = np.linspace(1.5, 2, 500)
chi_list2 = []
for i in range(0,500):
    n = chi_un(x[i])
    chi_list2.append(n)

plt.plot(x, chi_list2)
plt.grid()
plt.xlabel("Resistance [Ohm]")
plt.ylabel("Chi squared")
plt.title("Chi squared of resistance with uncertainty")
plt.savefig("2b_chi_squared_resistance_uncertainty.png")
#plt.show()
plt.clf()

min_chi_unc = min(chi_list2)

amin = []
for i in range(0,500):
    if min_chi_unc == chi_un(x[i]):
        amin_x = x[i]
        amin.append(amin_x)
        break


print("2b) The minimum value of chi squared is at "+ str(round(min_chi_unc, 2)) + " and the best fit value for R at 1.77 Ohm. ")

#2c)

plt.errorbar(v, a, yerr=unc_a, linestyle="None", color="gray")
plt.scatter(v, a, linestyle="None", color="black")
plt.plot(v, 1/1.77*v)
plt.grid()
plt.ylabel("Current [A]")
plt.xlabel("Voltage [V]")
label = ("best fit", "data", "errors")
plt.legend(label)
plt.title("current vs. voltage measurements with best fit line")
plt.savefig("2c_plot_overlay.png")
#plt.show()
plt.clf()
print("2c) The overlay works fine for larger values but does not perform well for the first two measurements. "
      "Since the errorbar from the first two measurements do not overlap with the best fit line, the best fit performs insufficiently there. Adding an nuisance parameter might help (offset, e_bias). ")


#2d)

x = np.linspace(1.5, 2, 500)
min_chi_unc_1 = min_chi_unc+1
for i in range(0,1000):
    if min_chi_unc+1 - chi_un(x[i]) >=-0.1 and min_chi_unc+1 - chi_un(x[i]) <= 0.1:
        s = x[i]
        break

sigma_R = abs(amin[0]-s)
print("2d) The result from the delta chi square rule is " + str(round(amin[0], 2)) +"+-"+str(round(sigma_R, 2))+
      " Ohm, which is not compatible with R = 2 Ohm. ")

#2e)
#define function with bias:
e_bias = 0.7 #ampere
x = np.linspace(1.5, 2, 500)
def chi_bias(r):
    X2_bias = 0
    for i in range(0,5):
        X2_bias = X2_bias + ((a[i]-v[i]/r - e_bias)**2)/(unc_a[i]**2)
    return X2_bias

#calculate all values of chi squared biased-> add to list
chi_list_bias = []
for i in range(0, 500):
    chi_list_bias.append(chi_bias(x[i]))

#find minimum of chi squared biased
chi_bias_min = min(chi_list_bias)

#find minimum-value for resistance:
r_min_bias = 0
for i in range(0,500):
    if chi_bias_min == chi_bias(x[i]):
        r_min_bias = x[i]
        break

#calculate delta chi squared biased + 1 to get uncertainty sigma
for i in range(0,500):
    if chi_bias_min+1 - chi_bias(x[i]) >=-0.1 and chi_bias_min+1 - chi_bias(x[i]) <= 0.1:
        s = x[i]
        break
sigma_r_bias = abs(r_min_bias-s)

print("2e) Now, the minimum value of chi squared (bias) is at 7. This corresponds to an R-value of " + str(round(r_min_bias, 3)) +"+-"+str(round(sigma_r_bias, 3))+
      " Ohm, where 2 Ohm lies within the error range. Therefore has become compatible with 2 Ohm by adding the nuisance parameter e_bias. ")

#2f)
ndf1 = 5
ndf2 = 4

gof1 = min_chi_unc/ndf1
gof2 = chi_bias_min/ndf2

p1 = chi2.cdf(min_chi_unc, ndf1)
p2 = chi2.cdf(chi_bias_min, ndf2)

gof_p1 = 1-p1
gof_p2 = 1-p2

print("2f) The goodness of fit without offset parameter is " +str(round(gof1, 2)) + "."+
      "The goodness of fit with offset parameter is " +str(round(gof2, 2)) + "."+
      "The goodness of fit probability without the offset parameter reads "+str(round(gof_p1, 3)) + "."+
      "The goodness of fit probability with the offset parameter reads "+str(round(gof_p2, 3)) + "."+
      "The goodness of fit varies quite strongly between non-biased and the biased (by a factor of 2), which shows that the e_bias can improve the fit by a lot. ")

#2g
def test(x, e, r):
    return e + x/r
para, para_covariance = optimize.curve_fit(test,v,a, p0=[2,2])
print("2g) The optimised parameters are e_bias = " + str(round(para[0] ,2)) + " and R = " + str(round(para[1],2)) + ".")

#2h
var_r = para_covariance[1][1]
unc_r = np.sqrt(var_r)
var_e = para_covariance[0][0]
unc_e = np.sqrt(var_e)
cov_re = para_covariance[0][1]
corr_coeff = cov_re/(unc_e*unc_r)

print("2h) The uncertainty calculated by the covariance matrix are for R = " + str(round(unc_r, 2)) +" and for e_bias = "+ str(round(unc_e, 2))+". "
    "The correlation coefficient reads " + str(round(corr_coeff,2)) +
    ". The uncertainty from 2e) was " + str(round(sigma_r_bias, 4)) +
    " and when we compare it to the uncertainty we obtained from the curve fit (" + str(round(unc_r, 3)) + ") we see that " \
    "the new value is actually larger by a factor of ~10.  " \
    "According to the lecture, it is possible to get a worse fit if X^2/n_f is smaller than 1 which is the case here. ")

