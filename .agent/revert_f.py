import os
F=102; N=110; I=105; E=101; R=114; Ncap=78
def B(*cs): return bytes(cs)
# true names: n i n n e r
NN = B(110,105,110,110,101,114)      # n i n n e r
NNcap = B(78,105,110,110,101,114)    # N i n n e r
# the f variant to revert
ninf = B(110,105,110,102,101,114)    # n i n f e r
Ninf = B(78,105,110,102,101,114)     # N i n f e r
p='p1v4_full_cycle.ps1'
b=open(p,'rb').read()
c1=b.count(Ninf); b=b.replace(Ninf, NNcap)
c2=b.count(ninf); b=b.replace(ninf, NN)
open(p,'wb').write(b)
print('Ninf->NNcap:',c1,'ninf->nn:',c2)
print('NNcap in file:', b.count(NNcap), ' nn:', b.count(NN))
