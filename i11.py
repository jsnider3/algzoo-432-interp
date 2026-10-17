import numpy as np
from seeds_clocks import load, run, T
for name,path,src,delay in [("mystery (n5)","weights.npz",5,1),("seed 2 (n10)","weights_seed2.npz",10,0)]:
    Wih,Whh,Wo=load(path)
    # impulse response columns for event ages 0..9 (pointer fires 'delay' steps after the event)
    R=[]
    for age in range(10):
        a=age-delay
        R.append(np.linalg.matrix_power(Whh,a)@np.eye(16)[src] if a>=0 else np.zeros(16))
    R=np.array(R).T                          # 16 x 10, column for age
    Y=np.eye(10)[:, ::-1]                    # slot 9-age
    keep=[a for a in range(10) if a>=delay]
    Wstar=Y[:,keep]@np.linalg.pinv(R[:,keep])        # min-norm decoder of the impulse responses
    c=(Wo*Wstar).sum()/np.linalg.norm(Wo)/np.linalg.norm(Wstar)
    rowc=[(Wo[i]@Wstar[i])/np.linalg.norm(Wo[i])/np.linalg.norm(Wstar[i]) for i in range(10)]
    x=np.random.default_rng(141).standard_normal((300000,T)); h=run(x,Wih,Whh)[:,-1]; tgt=np.argsort(x,1)[:,-2]
    acc_trained=((h@Wo.T).argmax(1)==tgt).mean()
    # swap readout: impulse-inverse decoder, refit only a per-slot scale+bias-free mixing with the trained readout? test raw first
    acc_star=((h@Wstar.T).argmax(1)==tgt).mean()
    print(f"{name}: cosine(trained Wo, impulse-inverse decoder) {c:.2f}; per-row {np.round(rowc,2)} | accuracy with trained Wo {acc_trained:.3f}, with impulse-inverse decoder {acc_star:.3f}")
