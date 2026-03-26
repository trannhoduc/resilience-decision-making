import PARAMETERS as P
from computation import precompute_all

derived = precompute_all(P)

print(derived["stationary"]["s_bar"])
print(derived["markov_surrogate"]["q01"])
print(derived["markov_chain_statistics"]["pi0"])