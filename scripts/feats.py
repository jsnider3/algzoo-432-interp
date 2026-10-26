from common import *
def features(x, t):
    """features at time t (state after reading x_0..x_t)"""
    F = {}
    X = x[:, :t+1]
    Z = np.concatenate([np.zeros((x.shape[0],1)), X], 1)
    F['M'] = Z.max(1)
    F['Mprev'] = Z[:, :-1].max(1)  # excl latest
    for k in range(t+1):
        F[f'x[t-{k}]'] = X[:, t-k]
        keep = [j for j in range(t+1) if j != t-k]
        F[f'loo{k}'] = np.concatenate([np.zeros((x.shape[0],1)), X[:, keep]],1).max(1)
    F['newmax'] = np.maximum(X[:, t] - F['Mprev'], 0)
    srt = np.sort(Z, 1)
    F['2nd'] = srt[:, -2] if Z.shape[1] > 1 else 0*F['M']
    return F
