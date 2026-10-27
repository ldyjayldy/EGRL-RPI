import torch
import esm
import urllib.request
import certifi
import ssl
import os

context = ssl.create_default_context(cafile=certifi.where())
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
urllib.request.install_opener(opener)

def load_esm2_model(model_name="esm2_t33_650M_UR50D"):
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    except AttributeError:
        try:
            model = torch.hub.load('facebookresearch/esm', model_name)
            alphabet = model.alphabet
        except Exception as e:
            from esm import pretrained
            model, alphabet = pretrained.load_model_and_alphabet(model_name)

    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def read_txt_file(file_path):
    protein_sequences = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 2:
                    protein_id = parts[0]
                    sequence = parts[1]
                    protein_sequences[protein_id] = sequence
    return protein_sequences


def encode_protein_sequences(model, batch_converter, protein_sequences):
    encoded_features = {}
    with torch.no_grad():
        for name, seq in protein_sequences.items():
            batch_labels, batch_strs, batch_tokens = batch_converter([(name, seq)])
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]

            sequence_representation = token_representations[:, 0, :].squeeze()
            encoded_features[name] = sequence_representation.cpu().numpy()
    return encoded_features

def save_encoded_features(encoded_features, output_file):
    with open(output_file, 'w') as f:
        for name, features in encoded_features.items():
            feature_str = ' '.join(map(str, features))
            f.write(f"{name} {feature_str}\n")


def main():
    try:

        print("正在加载ESM2模型...")
        model, batch_converter = load_esm2_model()
        print("模型加载成功")


        txt_file = "Data2_sequence.txt"  # 请根据实际情况修改路径
        print(f"正在读取蛋白质序列文件: {txt_file}")
        protein_sequences = read_txt_file(txt_file)
        print(f"成功读取 {len(protein_sequences)} 个蛋白质序列")


        print("正在对蛋白质序列进行编码...")
        encoded_features = encode_protein_sequences(model, batch_converter, protein_sequences)


        output_file = "Supp_A_encoded_protein_features.txt"  # 请根据实际情况修改路径
        save_encoded_features(encoded_features, output_file)

        print(f"蛋白质序列编码完成，编码结果已保存到 {output_file}")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请尝试以下解决方案:")
        print("1. 重新安装ESM库: pip install --upgrade fair-esm")
        print("2. 或者安装开发版本: pip install git+https://github.com/facebookresearch/esm.git")
        print("3. 检查Python环境是否正确")


if __name__ == "__main__":
    main()