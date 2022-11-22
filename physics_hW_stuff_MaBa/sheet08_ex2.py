import numpy as np

lamas = [656.3, 486.1, 434, 410.2, 397, 389]
m = np.linspace(3, 8, 6)

lamas_nm = []
for i in lamas:
    lamas_nm.append(i*10**(-9))

R = 1/(364.6*(10**-9)*(1/4))
for i, val in enumerate(lamas_nm):
    R += 1/(val*(1/4-1/(m[i]**2)))

print(R/7)
