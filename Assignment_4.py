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
from numpy.core.fromnumeric import argmax
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
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
    [0.90, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00],
    [0.05, 0.00, 0.00, 0.90, 0.05, 0.00, 0.00],
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

def HMM_3_translate(sequence:str)->str:
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
def HMM_7_translate(sequence:str)->str:
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
    init_probs = np.log(hmm.init_probs)
    trans_probs = np.log(hmm.trans_probs)
    emission_probs = np.log(hmm.emission_probs)
    
	# Initialise the first column of V
    for i in range(0, K):
        V[i][0] = init_probs[i] + emission_probs[i][X[0]]

     # Implement the Viterbi algorithm
    for i in range(1, N):
        if i % 100000 == 0:
            print(i)
        for j in range(0, K):
           t_probs =   V[:,i-1] + trans_probs[:,j]
           V[j][i] = max(t_probs) + emission_probs[j][X[i]] 
    print(V[:,0] == init_probs + emission_probs[:,X[0]])
    return V

#%% Backwards Alg:

def backwards(V, hmm):
    assert len(V) == len(hmm.init_probs)
    
    init_probs = np.log(hmm.init_probs)
    trans_probs = np.log(hmm.trans_probs)
    emission_probs = np.log(hmm.emission_probs)
    
    N = len(V[0])
    i = N - 1
    k = argmax(V[:,N-1])
    o = []
    
    while i >= 0:
        o.append(str(k))
        k = argmax(V[:,i-1] + trans_probs[:,k] )
        i-=1
        if i % 100000 == 0: print(i)
    return ''.join(o[::-1])
    
#%%

def compute_accuracy(true_ann, pred_ann):
    if len(true_ann) != len(pred_ann):
        return 0.0
    return sum(1 if true_ann[i] == pred_ann[i] else 0 
               for i in range(len(true_ann))) / len(true_ann)

#%%
'''
Finding Genes: 
'''
#%% Q1, genome 1, 7 state model.
dat = read_fasta_file("true-ann1.fa")
true = HMM_7_translate(dat["true-ann1"])

obs = read_fasta_file("genome1.fa")
tmp = viterbi(obs["genome1"], hmm_7_state)
pred = backwards(tmp, hmm_7_state)


compute_accuracy(true, pred)
#Resulted in 26.5% accuracy

#%% Q2, genome 2, 7 state model
dat = read_fasta_file("true-ann2.fa")
true = HMM_7_translate(dat["true-ann2"])

obs = read_fasta_file("genome2.fa")
tmp = viterbi(obs["genome2"], hmm_7_state)
pred = backwards(tmp, hmm_7_state)

compute_accuracy(true, pred)
#resulted in 30.3% accuracy

#%% Q3, genome 1, 3 state model
dat = read_fasta_file("true-ann1.fa")
true = HMM_3_translate(dat["true-ann1"])

obs = read_fasta_file("genome1.fa")
tmp = viterbi(obs["genome1"], hmm_3_state)
pred = backwards(tmp, hmm_3_state)

compute_accuracy(true, pred)
#resulted in 31.9% accuracy

#%% Q4, genome 2, 3 state model
dat = read_fasta_file("true-ann2.fa")
true = HMM_3_translate(dat["true-ann2"])

obs = read_fasta_file("genome2.fa")
tmp = viterbi(obs["genome2"], hmm_3_state)
pred = backwards(tmp, hmm_3_state)

compute_accuracy(true, pred)
#resUlted in 35.1% accuracy


#%%
'''
We make a function to train new models, so we can do it easily for each of the next tasks
We take no responsibility for any strokes induced by looking at the following code.
'''

def training_model(data, true_ann, hm)->hmm:
    obs = translate_observations_to_indices(data)
    N = len(obs)
    r = len(hm.init_probs)
    c = 4
    '''
    We calculate the emission probs by zipping the sequense and the true-ann 
    and count the nr of nucleotides occuring in each state. 
    Each count is then divded by row sum.
    '''

    emission = np.zeros((r,c))
    for state, nuc in zip(true_ann, obs):
        emission[int(state)][int(nuc)] +=1
    
    for i in range(0, r):
        div = sum(emission[i,:])
        for j in range(0, c):
            emission[i][j] /= div
    
    '''
    We calculate transitions by counting each instance of a transition in the true-ann
    and divide count by the sum of that row
    '''

    transition = np.zeros((r, r))
    cunt = 0
    nxt = cunt+1
    while cunt < N-2:
        transition[int(true_ann[cunt])][int(true_ann[nxt])] +=1
        cunt +=1
        nxt +=1
    
    for i in range(0, r):
        div = sum(transition[i,:])
        for j in range(0, r):
            transition[i][j] /= div
    
    '''
    We use the known initial state from true ann, 
    to set the initial prob, since there is one known possible start state
    we set that prob to 1
    '''
    initial = np.zeros((r))
    initial[int(true_ann[0])] = 1

    model = hmm(initial, transition, emission)
    return model
#%%

'''
Testing out the training model Q5: 
Using genome 1 to train, and analyzing genome 2. With 7 state model
'''
dat = read_fasta_file("true-ann1.fa")
true = HMM_7_translate(dat["true-ann1"])
obs = read_fasta_file("genome1.fa")

#Make trained model
hmm_7_trained = training_model(obs["genome1"], true, hmm_7_state)

#Load genome 2 and true-ann 2
obs2 = read_fasta_file("genome2.fa")

tmp = viterbi(obs2["genome2"], hmm_7_trained)
pred = backwards(tmp, hmm_7_trained)

dat = read_fasta_file("true-ann2.fa")
true = HMM_7_translate(dat["true-ann2"])

#compute accuracy
print(compute_accuracy(true, pred))
#resulted in 78% accuracy

#%%

'''
Testing out the training model Q6:
Using genome 1 to train, and analyzing genome 2. With 3 state model
'''
dat = read_fasta_file("true-ann1.fa")
true = HMM_3_translate(dat["true-ann1"])
obs = read_fasta_file("genome1.fa")

#Make trained model
hmm_3_trained = training_model(obs["genome1"], true, hmm_3_state)

#Load genome 2 and true-ann 2
obs2 = read_fasta_file("genome2.fa")

tmp = viterbi(obs2["genome2"], hmm_3_trained)
pred = backwards(tmp, hmm_3_trained)

dat = read_fasta_file("true-ann2.fa")
true = HMM_3_translate(dat["true-ann2"])

#Compute accuracy
print(compute_accuracy(true, pred))
#resulted in 57.4% accuracy


#%%

'''
Testing out the training model Q7: 
Using genome 2 to train, and analyzing genome 1. With 7 state model
'''
dat = read_fasta_file("true-ann2.fa")
true = HMM_7_translate(dat["true-ann2"])
obs = read_fasta_file("genome2.fa")

#Make trained model
hmm_7_trained = training_model(obs["genome2"], true, hmm_7_state)

#Load genome 1 and true-ann 1
obs2 = read_fasta_file("genome1.fa")

tmp = viterbi(obs2["genome1"], hmm_7_trained)
pred = backwards(tmp, hmm_7_trained)

dat = read_fasta_file("true-ann1.fa")
true = HMM_7_translate(dat["true-ann1"])

#compute accuracy
print(compute_accuracy(true, pred))
#resulted in 76% accuracy

#%%

'''
Testing out the training model Q8:
Using genome 2 to train, and analyzing genome 1. With 3 state model
'''
dat = read_fasta_file("true-ann2.fa")
true = HMM_3_translate(dat["true-ann2"])
obs = read_fasta_file("genome2.fa")

#Make trained model
hmm_3_trained = training_model(obs["genome2"], true, hmm_3_state)

#Load genome 1 and true-ann 1
obs2 = read_fasta_file("genome1.fa")

tmp = viterbi(obs2["genome1"], hmm_3_trained)
pred = backwards(tmp, hmm_3_trained)

dat = read_fasta_file("true-ann1.fa")
true = HMM_3_translate(dat["true-ann1"])

#Compute accuracy
print(compute_accuracy(true, pred))
#resulted in 59.1% accuracy



#%%

'''
Code testing area below:
'''
#%%





#%%
dat = read_fasta_file("true-ann1.fa")
#%%
print(HMM_3_translate(dat["true-ann1"][200:400]))
# %%
print(HMM_7_translate(dat["true-ann1"][200:400]))
# %%
dat = read_fasta_file("genome1.fa")
#%%


print(viterbi(dat["genome1"][0:10],hmm_3_state))
# %%
derp = viterbi(dat["genome1"][:],hmm_7_state)

# %%
pred = backwards(derp, hmm_7_state)
#%%
reeeee = read_fasta_file("true-ann1.fa")
actual = HMM_7_translate(reeeee["true-ann1"][:])

#%%
print(len(actual), len(pred))
compute_accuracy(actual, pred)
# %%
