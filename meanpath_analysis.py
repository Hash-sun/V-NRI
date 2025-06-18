import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import argparse
import networkx as nx
#import ipysigma as ips

parser = argparse.ArgumentParser(
    'Find mean shortest paths along domains in residues.')
parser.add_argument('--num-residues', type=int, default=220,
                    help='Number of residues of the PDB.')
parser.add_argument('--source-node', type=int, default=172,
                    help='source residue of the PDB')
parser.add_argument('--outputfilename', type=str, default='data/no_end_source.txt',
                    help='File of shortest path from source to targets')
parser.add_argument('--edges_a', type=str, default='data/pdb_a/logs/filtered_edges_12.npy',
                    help='File of edge weights for pdb_a')
parser.add_argument('--edges_b', type=str, default='data/pdb_b/logs/filtered_edges_12.npy',
                    help='File of edge weights for pdb_b')
args = parser.parse_args()

def process(number):
    if number < 198:
        result = str(number + 94)
    else:
        dg_mapping = {
            198: 'DG2', 199: 'DG3', 200: 'DG4', 201: 'DC5', 202: 'DA6', 203: 'DT7',
            204: 'DG8', 205: 'DC9', 206: 'DC10', 207: 'DC11', 208: 'DG12',
            209: 'DG2\'', 210: 'DG3\'', 211: 'DG4\'', 212: 'DC5\'', 213: 'DA6\'', 214: 'DT7\'',
            215: 'DG8\'', 216: 'DC9\'', 217: 'DC10\'', 218: 'DC11\'', 219: 'DG12\''
        }
        result = dg_mapping.get(number, str(number))

    return str(result)

# edges_a = np.load('data/logs/filtered_edges_12.npy')
# edges_b = np.load('data/logs/filtered_edges_12.npy')

edges = (args.edges_a + args.edges_b) / 2
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






