F=102
N=110
def B(*cs): return bytes(cs)
# Target folder (N i n n e r) and process (n i n n e r)
Nfolder=B(78,105,110,110,101,114)   # N i n n e r
nproc=B(110,105,110,110,101,114)    # n i n n e r
# Wrong variants to replace
Nf=B(78,105,110,102,101,114)        # N i n f e r
nf=B(110,105,110,102,101,114)       # n i n f e r

p='p1v4_full_cycle.ps1'
b=open(p,'rb').read()
# Folder: N f e r  -> N n e r   (only where followed by " AI")
b=b.replace(Nf+B(32,65,73), Nfolder+B(32,65,73))
# Process/exe: n f e r - s e r v e  -> n n e r - s e r v e
b=b.replace(nf+B(45,115,101,114,118,101), nproc+B(45,115,101,114,118,101))
open(p,'wb').write(b)
print("folder",Nfolder,"proc",nproc)
print("remaining Nf:",b.count(Nf),"nf:",b.count(nf))
