#Running the readme examples from https://github.com/quantumgizmos/bias_tailored_qldpc/blob/main/README.ipynb
from ldpc import protograph as pt
from ldpc.code_util import compute_code_distance
from lifted_hgp import lifted_hgp
from bposd.hgp import hgp
from bposd.css import css_code
import ldpc.mod2 as mod2
from ldpc.code_util import get_code_parameters
import numpy as np
a1=pt.array([
        [(0), (11), (7), (12)],
        [(1), (8), (1), (8)],
        [(11), (0), (4), (8)],
        [(6), (2), (4), (12)]])
L=3
ring_element=pt.RingOfCirculantsF2((0,1))
print(ring_element)
ring_element.to_binary(3)
L=5

a=pt.permutation_matrix(L,2)+pt.permutation_matrix(L,4)
b=pt.permutation_matrix(L,3)

#print((a@b %2+(a+b)%2)%2)
L=5

a=pt.RingOfCirculantsF2((2,4))
b=pt.RingOfCirculantsF2((3))
c= a*b + (a+b)

#print("Ring element")
#print(c)

#print()

#print("Matrix representation")
#print(c.to_binary(L))
a1=pt.array([
        [(0), (11), (7), (12)],
        [(1), (8), (1), (8)],
        [(11), (0), (4), (8)],
        [(6), (2), (4), (12)]])

#print(a1)
H=a1.to_binary(lift_parameter=13)
n,k,d,_,_=get_code_parameters(H)
#print(f"Code parameters: [{n},{k},{d}]")
qcode=hgp(H,H,compute_distance=False) #Set distance = true if you want it. it blocks execution
#qcode.test()
rate=qcode.K/qcode.N
rate
quantum_protograph_code=lifted_hgp(lift_parameter=13,a=a1,b=a1)
#print(quantum_protograph_code.hz_proto.__compact_str__())
hx=quantum_protograph_code.hx_proto.to_binary(lift_parameter=13)
hx
# Check weight check - x
print("X check weights ")
print(np.sum(hx,axis=1))
hz=quantum_protograph_code.hz_proto.to_binary(lift_parameter=13)
hz
print("Z check weights ")
print(np.sum(hz,axis=1))
qcode=css_code(hx,hz)
#qcode.test()
qcode=lifted_hgp(lift_parameter=13,a=a1,b=a1)
qcode.test()
