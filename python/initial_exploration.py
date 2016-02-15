# Illustris

# Look at galaxies that, in at least 1 out of 4 orientations, were classified as an edge-on disk.
# What did the other orientations show about that galaxy? Were they consistent?

# Should be able to re-use my gz_class code for this.

# To simplify, I'll start with galaxies with a single "set" of backgrounds. 

import pandas as pd
import numpy as np
from gz_class import plurality

df = pd.read_csv('collated_fixed_mass.csv')
n_unique = len((df['subhalo_id']).unique())

b0 = df[df['background'] == 0]
grouped = b0.groupby(['subhalo_id'])

fraccols = []
for c in b0.columns:
    if c[-4:] == 'frac':
        fraccols.append(c)

d = {1:0,2:0,3:0,4:0}
for name,group in grouped:

    votearr = np.array(group[fraccols])
    answers = []
    for v in votearr:
        e,a = plurality(v,'illustris') 
        answers.append(np.array(a)[np.array(e) == 1])
    answers_hashable = map(tuple, answers)
    d[len(set(answers_hashable))] += 1

from matplotlib import pyplot as plt

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.scatter(d.keys(),d.values(),s=40)
ax.set_xlim(0,5)
ax.set_xlabel('Number of different GZ classifications',fontsize=16)
ax.set_ylabel('Count',fontsize=20)
ax.set_title("Illustris - fixed_mass")

plt.savefig('initial_exploration_diffs.png')

# Check the edge-on disks specifically

edgeon_atleast_one = []
for name,group in grouped:

    votearr = np.array(group[fraccols])
    edgeon = 0
    for v in votearr:
        e,a = plurality(v,'illustris') 
        answers = np.array(a)[np.array(e) == 1]
        if 3 in answers:
            edgeon += 1

    if edgeon == 2:
        print group['zooniverse_id']
    if edgeon >= 1:
        edgeon_atleast_one.append(edgeon)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.hist(edgeon_atleast_one,bins=range(6))
ax.set_xlim(0,5)
ax.set_xlabel('Number of views with an edge-on disk',fontsize=16)
ax.set_ylabel('Count',fontsize=20)
ax.set_title("Illustris - fixed_mass edge-on")

plt.savefig('initial_exploration_edgeon.png')


    #if group['t00_smooth_or_features_a1_features_frac'] > 0.75 & group['t01_disk_edge_on_a0_yes_frac'] > 0.75:


# Pick
