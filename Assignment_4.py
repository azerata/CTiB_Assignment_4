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
