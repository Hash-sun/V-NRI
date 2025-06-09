import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import networkx as nx
#import ipysigma as ips

parser = argparse.ArgumentParser(
    'Find shortest paths along domains in residues.')
parser.add_argument('--num-residues', type=int, default=220,
                    help='Number of residues of the PDB.')
parser.add_argument('--source-node', type=int, default=172,
                    help='source residue of the PDB')
parser.add_argument('--outputfilename', type=str, default='data/p53-g266r/mean/4-Atten/no_end_source172-pro.txt',
                    help='File of shortest path from source to targets')
args = parser.parse_args()

def process(number):
    if number < 198:
        result = str(number + 94)
    else:
        dg_mapping = {
            198: 'DG2', 199: 'DG3', 200: 'DG4', 201: 'DG5', 202: 'DG6', 203: 'DG7',
            204: 'DG8', 205: 'DG9', 206: 'DG10', 207: 'DG11', 208: 'DG12',
            209: 'DG2\'', 210: 'DG3\'', 211: 'DG4\'', 212: 'DG5\'', 213: 'DG6\'', 214: 'DG7\'',
            215: 'DG8\'', 216: 'DG9\'', 217: 'DG10\'', 218: 'DG11\'', 219: 'DG12\''
        }
        result = dg_mapping.get(number, str(number))

    return str(result)

edges_a = np.load('data/p53-g266r/trj1/4-Atten/filtered_edges_12.npy')
edges_b = np.load('data/p53-g266r/trj2/4-Atten/new/filtered_edges_12.npy')

edges = (edges_a + edges_b) / 2
edges_list = list()
# Default: i->j
for i in range(args.num_residues):
    for j in range(args.num_residues):
        if i != j:
            edges_list.append((i, j, {'weight': edges[j, i]}))
        # print(edges_list)
MDG = nx.MultiDiGraph()
MDG.add_edges_from(edges_list)

source_node = args.source_node  # set source node

out_file = args.outputfilename

# Initialize variables for path selection
current_node = source_node
path = [source_node]
visited_nodes = set([source_node])
path_info = []
outedge = []
while True:
    # Get all outgoing edges from the current node
    outgoing_edges = MDG.out_edges(current_node, data=True)
    # print(outgoing_edges)
    for edge in outgoing_edges:
        if edge[2]['weight'] != 1:
            outedge.append(edge)

    if not outgoing_edges or all(edge[2]['weight'] == 1 for edge in outgoing_edges):
        break

    # Find the edge with the minimum weight
    min_weight = float('inf')
    min_edge = None
    for edge in outgoing_edges:
        if edge[2]['weight'] < min_weight and edge[1] not in visited_nodes:
            min_weight = edge[2]['weight']
            min_edge = edge

    if min_edge is None:
        break

    next_node = min_edge[1]
    path.append(next_node)
    visited_nodes.add(next_node)

    # Save path information
    path_info.append('-'.join(list(map(process, path))) + ' : ' + str(min_weight))

    # Update the current node
    current_node = next_node

# Write the results to the output file
with open(out_file, 'w') as f:
    f.write('Source node: %s\n' % process(source_node))
    for info in path_info:
        f.write('\t' + info + '\n')






