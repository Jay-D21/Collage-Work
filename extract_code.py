import json

with open('C227_Siamese_Network_using_Triplet_loss 1.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        print(''.join(cell['source']))
        print('\n# ---\n')
