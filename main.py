import numpy as np

c = 299792458
tau = 2e-6
s = 3e4

res = np.sqrt(1-((c**2)*(tau**2))/(s**2))
print(res)


