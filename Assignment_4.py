#%%
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
sæk = "NNNNNNNNNNNNNCCCCRNNNNNNRRRRRRRRCCCCCCCNNNNNNNNN"

print(HMM_translate(sæk))
# %%
dat = read_fasta_file("true-ann1.fa")
#%%
print(HMM_translate(dat["true-ann1"][500:550]))
# %%
