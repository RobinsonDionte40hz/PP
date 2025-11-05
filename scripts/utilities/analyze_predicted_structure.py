#!/usr/bin/env python3
"""Analyze predicted structure geometry to understand high RMSD."""

import numpy as np
from Bio.PDB import PDBParser

# Load predicted structure
parser = PDBParser(QUIET=True)
pred = parser.get_structure('pred', 'results/predicted_structures/1VII_predicted_octahedron.pdb')

# Extract CA coordinates
pred_ca = np.array([
    atom.coord 
    for model in pred 
    for chain in model 
    for res in chain 
    if 'CA' in res 
    for atom in [res['CA']]
])

# Calculate backbone vectors and angles
vectors = pred_ca[1:] - pred_ca[:-1]
norms = np.linalg.norm(vectors, axis=1)
normalized = vectors / norms[:, np.newaxis]
dots = np.sum(normalized[:-1] * normalized[1:], axis=1)
angles_deg = np.degrees(np.arccos(np.clip(dots, -1, 1)))

print('\n' + '='*70)
print('PREDICTED STRUCTURE GEOMETRY ANALYSIS')
print('='*70)

print(f'\nBACKBONE LINEARITY:')
print(f'  Bond length range: {norms.min():.2f} - {norms.max():.2f} Å')
print(f'  Mean bond angle: {angles_deg.mean():.1f}° (180° = perfectly straight)')
print(f'  Angle std dev: {angles_deg.std():.1f}°')
print(f'  Angles > 170°: {np.sum(angles_deg > 170)}/{len(angles_deg)} ({100*np.sum(angles_deg > 170)/len(angles_deg):.1f}%)')

print(f'\nINTERPRETATION:')
if angles_deg.mean() > 170:
    print('  ⚠️  Structure is NEARLY STRAIGHT (minimal bending)')
elif angles_deg.mean() > 150:
    print('  ⚠️  Structure is MOSTLY EXTENDED (slight bending)')
else:
    print('  ✓ Structure has significant backbone curvature')

print(f'\nFIRST 10 COORDINATES (Å):')
for i in range(min(10, len(pred_ca))):
    x, y, z = pred_ca[i]
    print(f'  Residue {i+1}: ({x:7.2f}, {y:7.2f}, {z:7.2f})')

print(f'\nLAST 10 COORDINATES (Å):')
for i in range(max(0, len(pred_ca)-10), len(pred_ca)):
    x, y, z = pred_ca[i]
    print(f'  Residue {i+1}: ({x:7.2f}, {y:7.2f}, {z:7.2f})')

# Check if structure is primarily along one axis
variance = np.var(pred_ca, axis=0)
print(f'\nVARIANCE BY AXIS:')
print(f'  X-axis: {variance[0]:.2f} Å²')
print(f'  Y-axis: {variance[1]:.2f} Å²')
print(f'  Z-axis: {variance[2]:.2f} Å²')
max_var_axis = ['X', 'Y', 'Z'][np.argmax(variance)]
print(f'  Primary extension: {max_var_axis}-axis ({np.max(variance)/np.sum(variance)*100:.1f}% of total variance)')

print('='*70 + '\n')
