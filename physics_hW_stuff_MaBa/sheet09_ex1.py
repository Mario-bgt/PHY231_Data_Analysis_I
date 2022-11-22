import numpy as np

wink = [54, 41, 35, 31, 28, 24]
winkels = []
for i in wink:
    x = (i/180)*np.pi
    winkels.append(x)

spanung = [9, 14, 18, 23, 28, 37]
lama = []
p = []
h = []
d = 0.25*10**(-9)
e = 1.6021*10**(-19)
m_e = 9.10938*10**(-31)

for j in range(len(winkels)):
    lamsa = np.sin(winkels[j])*d*2
    pep = np.sqrt(e*m_e*2*spanung[j])
    hebs = lamsa*pep
    lama.append(lamsa)
    p.append(pep)
    h.append(hebs)


print("Winkel: "+ str(winkels))
print("Spannung: " + str(spanung))
print("Lamda: "+ str(lama))
print("Impuls: "+ str(p))
print("H: " + str(h))
