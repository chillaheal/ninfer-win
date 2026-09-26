import os
B = lambda *cs: bytes(cs)
p = 'p1v4_full_cycle.ps1'
b = open(p, 'rb').read()

# mangled "ninfner-serve" (n i n f n e r -) -> "ninner-serve" (n i n f e r -)
bad1 = B(110,105,110,102,110,101,114,45,115,101,114,118,101)
good1 = B(110,105,110,102,101,114,45,115,101,114,118,101)
n1 = b.count(bad1); b = b.replace(bad1, good1)

# model ext  . n i n n e r  -> . n i n f e r
bad2 = B(46,110,105,110,110,101,114)
good2 = B(46,110,105,110,102,101,114)
n2 = b.count(bad2); b = b.replace(bad2, good2)

open(p, 'wb').write(b)
print('serve-mangle replaced:', n1)
print('model-ext replaced:', n2)
print('remaining ninfner:', b.count(B(110,105,110,102,110,101,114)))
