import os
out = open('qm9_smiles.csv', "w")
out.write(','.join(['smiles', 'tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']) + '\n')
for filename in os.listdir('qm9xyz'):
    with open('qm9xyz/'+filename, "r") as file:
        for line_number,line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                tag, index, A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv = line.split()
#             elif line_number in range(2:2+num_atoms):
#                 atomic_symbol, x, y, z, partial_charge = line.split()
#                 atomic_symbols.append(atomic_symbol)
#                 xyz_coordinates.append([float(x),float(y),float(z)])
            elif line_number == 3+num_atoms:
                smiles = line.split()[1]
    out.write(','.join([smiles, tag, index, A, B, C, mu, alpha, homo, lumo, gap, r2, zpve, U0, U, H, G, Cv])+'\n')
