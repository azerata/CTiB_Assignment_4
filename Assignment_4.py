#%%
'''
Thanks for the fasta parser:
'''
def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences
#%%
'''
Functions and probabilities from assignment:
'''
import numpy as np
def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]

def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)

class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

init_probs_3_state = np.array(
    [0.00, 0.10, 0.00]
)

trans_probs_3_state = np.array([
    [0.90, 0.10, 0.00],
    [0.05, 0.90, 0.05],
    [0.00, 0.10, 0.90],
])

emission_probs_3_state = np.array([
    #   A     C     G     T
    [0.40, 0.15, 0.20, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.20, 0.40, 0.30, 0.10],
])

hmm_3_state = hmm(init_probs_3_state,
                  trans_probs_3_state,
                  emission_probs_3_state)


init_probs_7_state = np.array(
    [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]
)

trans_probs_7_state = np.array([
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.90, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.05, 0.90, 0.05, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00],
])

emission_probs_7_state = np.array([
    #   A     C     G     T
    [0.30, 0.25, 0.25, 0.20],
    [0.20, 0.35, 0.15, 0.30],
    [0.40, 0.15, 0.20, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.20, 0.40, 0.30, 0.10],
    [0.30, 0.20, 0.30, 0.20],
    [0.15, 0.30, 0.20, 0.35],
])

hmm_7_state = hmm(init_probs_7_state,
                  trans_probs_7_state,
                  emission_probs_7_state)

import math
def log(x):
    if x == 0:
        return float("-inf")
    else:
        return math.log(x)
#%%
'''
Translation of 3 state model:
'''

def HMM_translate(sequence:str)->str:
    o = []
    for chr in sequence:
        match chr:
            case "C":
                o.append("0")
            case "N":
                o.append("1")
            case "R":
                o.append("2")
    return "".join(o)

#%%
'''
Translation of 7 state model:
'''
def HMM_7_State(sequence:str)->str:
    o = []
    c_streak = 0
    r_streak = 0
    for chr in sequence:
        match chr:
            case "N":
                o.append("3")
                c_streak = 0
                r_streak = 0
            case "C":
                o.append(str(c_streak % 3))
                c_streak +=1
                r_streak = 0
            case "R":
                o.append(str( (r_streak % 3) + 4 ))
                r_streak += 1
                c_streak = 0
    return "".join(o)


#%%

'''
Viterbi Implementation:
'''
def viterbi(obs, hmm):
    X = translate_observations_to_indices(obs)
    N = len(X)
    K = len(hmm.init_probs)
    V = np.zeros((K,N))
    hmm.init_probs = log(hmm.init_probs)
    hmm.trans_probs = log(hmm.trans_probs)
    hmm.emission_probs = log(hmm.emission_probs)

	# Initialise the first column of V
    for i in range(0, K):
        V[i][0] = hmm.init_probs[i] + hmm.emission_probs[i][X[0]]
    init_prop = max(V[:,[0]])
    print(init_prop)

    assert V[:,0] == hmm.init_probs + hmm.emission_probs[:,[X[0]]]
     # Implement the Viterbi algorithm
    #for i in range(1, N):
    #    for j in range(0, K):
    #        t_probs =   V[:,[i-1]] + log(hmm.trans_probs[:,[j]])
    #        print(t_probs)
            #V[j][i] = max(V[:,[i-1]]



    return V


'''
Code testing area below:
'''
#%%
dat = read_fasta_file("true-ann1.fa")
#%%
print(HMM_translate(dat["true-ann1"][200:400]))
# %%
print(HMM_7_State(dat["true-ann1"][200:400]))
# %%
dat = read_fasta_file("genome1.fa")
#%%


print(viterbi(dat["genome1"][0:10],hmm_3_state))
# %%
