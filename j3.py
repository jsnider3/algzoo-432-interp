import numpy as np
from seeds_clocks import load, run, T
from j2 import selectivity
A=load("ckpt/s2_e24.npz"); B=load("ckpt/s2_e25.npz"); n=10
x=np.random.default_rng(181).standard_normal((200000,T)); y=np.argsort(x,1)[:,-2]
def acc(Wih,Whh,Wo): return ((run(x,Wih,Whh)[:,-1]@Wo.T).argmax(1)==y).mean()
def mix(base,donor,rows_ih=(),rows_hh=(),cols_hh=()):
    Wih,Whh,Wo=[a.copy() for a in base]
    for r in rows_ih: Wih[r]=donor[0][r]
    for r in rows_hh: Whh[r]=donor[1][r]
    return Wih,Whh,Wo
others=[i for i in range(16) if i!=n]
cases={
 "2^24 as is":A, "2^25 as is":B,
 "2^24 + n10's incoming row from 2^25":mix(A,B,[n],[n]),
 "2^24 + all OTHER incoming rows from 2^25 (n10 row kept)":mix(A,B,others,others),
 "2^24 + Wih only from 2^25":mix(A,B,range(16),[]),
 "2^24 + Whh only from 2^25":mix(A,B,[],range(16)),
}
for k,(Wih,Whh,Wo) in cases.items():
    print(f"{k:58s} n10 selectivity {selectivity(Wih,Whh,n)}")
