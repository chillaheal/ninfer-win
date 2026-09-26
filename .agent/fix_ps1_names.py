import os
F = 102  # f
N = 110  # n
NIN = bytes([78, 105, 110])
NIN = bytes([78, 105, 110])
nin = bytes([110, 105, 110])
fch = bytes([F])
NIN = NIN + fch + bytes([101, 114])      # N i n f e r
NIN = NIN + bytes([101, 114])
nin = nin + fch + bytes([101, 114])
nin = nin + bytes([101, 114])
NL = chr(92)
root = 'C:' + NL + 'Users' + NL + 'Micke' + NL + 'Documents' + NL + 'Kodprojekt' + NL + NIN.decode() + ' Flash'
desk = 'C:' + NL + 'Users' + NL + 'Micke' + NL + 'Desktop'
desk = desk + NL + NIN.decode()
desk = desk + ' AI'
p = 'C:/Users/Micke/Documents/Kodprojekt/' + NIN.decode() + ' Flash/.agent/p1v4_full_cycle.ps1'
b = open(p, 'rb').read()
# 1. root
b = b.replace(bytes([78,105,110]) + bytes([110,101,114]), NIN, 1)
# 2. desktop folder: N i n n e r   A I  -> N i n f e r   A I
folder = NIN + b' AI'
b = b.replace(bytes([78,105,110]) + bytes([110,101,114]) + b' AI', folder)
# 3. process name n i n n e r - s e r v e  ->  n i n f e r - s e r v e
b = b.replace(bytes([110,105,110]) + bytes([110,101,114]) + b'-serve', nin + b'-serve')
open(p, 'wb').write(b)
print('OK')
print('root', NIN.decode())
print('desk', NIN.decode())
print('proc', nin.decode())
