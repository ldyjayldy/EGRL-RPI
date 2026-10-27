import os
from collections import Counter
import torch
from Bio import SeqIO
from sklearn.model_selection import train_test_split


def shift_prot_edges(edge, offset):
    if edge.numel() == 0:
        return edge
    return edge + offset


def load_sequences_from_fasta(fasta_path):
    sequences = {}
    if not os.path.exists(fasta_path):
        return sequences
    with open(fasta_path) as f:
        for record in SeqIO.parse(f, "fasta"):
            sequences[record.id] = str(record.seq).upper()
    return sequences


def load_encodings_from_file(enc_path, default_dim=128):
    if not os.path.exists(enc_path):
        return None
    encodings = {}
    with open(enc_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            sid = parts[0]
            vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
            encodings[sid] = vec
    return encodings


def load_interactions(interaction_path):
    triples = []
    if not os.path.exists(interaction_path):
        return triples
    with open(interaction_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                protein, rna, label = parts[0], parts[1], parts[2]
                try:
                    label = int(label)
                except:
                    continue
                triples.append((rna, protein, label))
    return triples


def safe_train_test_split(triples, test_size, random_state, stratify=True):
    labels = [t[2] for t in triples]
    if stratify and len(set(labels)) > 1:
        cnt = Counter(labels)
        if min(cnt.values()) >= 2:
            try:
                return train_test_split(
                    triples, test_size=test_size,
                    stratify=labels, random_state=random_state, shuffle=True,
                )
            except ValueError:
                pass
    return train_test_split(
        triples, test_size=test_size,
        random_state=random_state, shuffle=True,
    )
