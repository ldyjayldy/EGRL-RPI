import torch
import esm
import fm
import urllib.request
import certifi
import ssl
import os

context = ssl.create_default_context(cafile=certifi.where())
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
urllib.request.install_opener(opener)


# ==================== Model loading ====================

def load_esm2_model(model_name="esm2_t33_650M_UR50D"):
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    except AttributeError:
        try:
            model = torch.hub.load('facebookresearch/esm', model_name)
            alphabet = model.alphabet
        except Exception:
            from esm import pretrained
            model, alphabet = pretrained.load_model_and_alphabet(model_name)

    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def load_rna_fm_model():
    """Load RNA-FM"""
    model, alphabet = fm.pretrained.rna_fm_t12()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


# ==================== File reading ====================

def read_fasta_file(file_path):
    """Read sequences in FASTA format"""
    sequences = {}
    seq_id = None
    seq_chunks = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if seq_id is not None:
                    sequences[seq_id] = ''.join(seq_chunks)
                seq_id = line[1:].split()[0].replace('|', '_')
                seq_chunks = []
            elif seq_id is not None:
                seq_chunks.append(line)
    if seq_id is not None:
        sequences[seq_id] = ''.join(seq_chunks)
    return sequences


def read_txt_file(file_path):
    protein_sequences = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    protein_sequences[parts[0]] = parts[1]
    return protein_sequences

def truncate_sequences(sequences, max_len):
    truncated = 0
    out = {}
    for name, seq in sequences.items():
        if len(seq) > max_len:
            truncated += 1
            out[name] = seq[:max_len]
        else:
            out[name] = seq
    if truncated:
        print(f"  [Warning] {truncated} sequences exceed the maximum length {max_len} and will be truncated.")
    return out


# ==================== Encoding functions ====================

def encode_protein_sequences(model, batch_converter, protein_sequences):
    """Encode protein sequences with ESM2"""
    encoded_features = {}
    with torch.no_grad():
        for name, seq in protein_sequences.items():
            _, _, batch_tokens = batch_converter([(name, seq)])
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            sequence_representation = token_representations[:, 0, :].squeeze()
            encoded_features[name] = sequence_representation.cpu().numpy()
    return encoded_features


def encode_rna_sequences(model, batch_converter, rna_sequences):
    """Encode RNA sequences with RNA-FM"""
    encoded_features = {}
    with torch.no_grad():
        for name, seq in rna_sequences.items():
            _, _, batch_tokens = batch_converter([(name, seq)])
            results = model(batch_tokens, repr_layers=[12])
            token_representations = results["representations"][12]
            sequence_representation = token_representations[:, 0, :].squeeze()
            encoded_features[name] = sequence_representation.cpu().numpy()
    return encoded_features


# ==================== Saving ====================

def save_encoded_features(encoded_features, output_file):
    with open(output_file, 'w') as f:
        for name, features in encoded_features.items():
            feature_str = ' '.join(map(str, features))
            f.write(f"{name} {feature_str}\n")


# ==================== Main workflow ====================

def main():
    try:
        # ---------- 1. Protein encoding ----------
        print("Loading ESM2 model...")
        protein_model, protein_batch_converter = load_esm2_model()
        print("ESM2 model loaded successfully")

        # FASTA format
        protein_file = "Data2_sequence.fasta"
        print(f"Reading protein sequence file: {protein_file}")
        protein_sequences = read_fasta_file(protein_file)
        protein_sequences = truncate_sequences(protein_sequences, max_len=1022)
        print(f"Successfully read {len(protein_sequences)} protein sequences")

        print("Encoding protein sequences...")
        protein_features = encode_protein_sequences(
            protein_model, protein_batch_converter, protein_sequences
        )

        protein_output = "Dataset_protein_encoded"
        save_encoded_features(protein_features, protein_output)
        print(f"Protein sequence encoding completed; results saved to {protein_output}")

        # ---------- 2. RNA encoding ----------
        print("\nLoading RNA-FM model...")
        rna_model, rna_batch_converter = load_rna_fm_model()
        print("RNA-FM model loaded successfully")

        rna_file = "RNA_sequence.fasta"
        print(f"Reading RNA sequence file: {rna_file}")
        rna_sequences = read_fasta_file(rna_file)

        rna_sequences = truncate_sequences(rna_sequences, max_len=1022)
        print(f"Successfully read {len(rna_sequences)} RNA sequences")

        print("Encoding RNA sequences...")
        rna_features = encode_rna_sequences(
            rna_model, rna_batch_converter, rna_sequences
        )

        rna_output = "Dataset_RNA_encoded"
        save_encoded_features(rna_features, rna_output)
        print(f"RNA sequence encoding completed; results saved to {rna_output}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check the following:")
        print("1. Whether ESM2 dependencies are installed: pip install fair-esm")
        print("2. Whether RNA-FM is installed: pip install rna-fm")
        print("3. Whether input files are in FASTA format and paths are correct")
        print("4. Whether sequence IDs are unique (the first field of each FASTA header must be distinct)")


if __name__ == "__main__":
    main()