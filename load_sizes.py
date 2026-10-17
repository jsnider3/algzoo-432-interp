import numpy as np
from alg_zoo import zoo_2nd_argmax
for hsz in [4,6,8,32]:
    m=zoo_2nd_argmax(hsz,10); np.savez(f"work/weights_h{hsz}.npz", **{k:v.numpy() for k,v in m.state_dict().items()}); print(hsz,"ok")
