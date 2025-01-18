import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import os

def plot_scores(scores):
    plt.subplot(2,1,1)
    plt.plot(scores)
    plt.ylim([0,1])
    plt.plot([950, 950, float('nan'), 1050, 1050],
            [0,1,float('nan'),0,1], color='k', label='Zoom')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(scores)
    plt.xlim([950, 1050])
    plt.ylim([0,1])
    

def plot_positional_encoding(positional_encoding, positional_encoding_2=None, positions=None, d_factors=None, length=None, depth=None):
    if positions is None:
        if length is None:
            plt.figure(figsize=(10, 6))
            plt.imshow(positional_encoding.T, aspect='auto', cmap='viridis', origin='lower')
            plt.title('Positional Encodings')
            plt.xlabel('Position')
            plt.ylabel('Depth')
            plt.yticks(np.arange(0, 4), ['0', '1', '2', '3'])
            plt.colorbar()
            plt.show()

        else:
            pos_encoding = positional_encoding(length, depth)
            plt.figure(figsize=(10, 6))
            plt.imshow(pos_encoding.T, aspect='auto', cmap='viridis', origin='lower')
            plt.title('Positional Encodings')
            plt.xlabel('Position')
            plt.ylabel('Depth')
            plt.colorbar()
            plt.show()

    else:
        if d_factors is None:
            if positional_encoding_2 is None:
                pos_encoding_discrete = positional_encoding(positions)
                plt.figure(figsize=(10, 6))
                plt.imshow(pos_encoding_discrete.T, aspect='auto', cmap='viridis', origin='lower')
                plt.title('Positional Encodings')
                plt.xlabel('Position')
                plt.ylabel('Depth')
                plt.yticks(np.arange(0, 4), ['0', '1', '2', '3'])
                plt.colorbar()
                plt.show()

            else:
                pos_encoding_discrete = positional_encoding_2(positions)
                pos_encoding_continuous = positional_encoding(positions)

                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                img1 = axes[0].imshow(pos_encoding_discrete.T, aspect='auto', cmap='viridis', origin='lower')
                axes[0].set_title('Positional Encodings Discrete')
                axes[0].set_xlabel('Position')
                axes[0].set_ylabel('Depth')
                axes[0].set_yticks(np.arange(0, 4))
                axes[0].set_yticklabels(['0', '1', '2', '3'])
                axes[0].grid(False)
                axes[0].set_aspect('auto')
                axes[0].set_xticks(np.arange(0, 16))
                plt.colorbar(img1, ax=axes[0])

                img2 = axes[1].imshow(pos_encoding_continuous.T, aspect='auto', cmap='viridis', origin='lower')
                axes[1].set_title('Positional Encodings Continuous')
                axes[1].set_xlabel('Position')
                axes[1].set_ylabel('Depth')
                axes[1].set_yticks(np.arange(0, 4))
                axes[1].set_yticklabels(['0', '1', '2', '3'])
                axes[1].grid(False)
                axes[1].set_aspect('auto')
                axes[1].set_xticks(np.arange(0, 16))
                plt.colorbar(img2, ax=axes[1])

                plt.tight_layout()
                plt.show()

        else:
            plt.figure(figsize=(12, 10))

            # Find the global minimum and maximum values across all plots
            global_min = float('inf')
            global_max = float('-inf')

            for d in d_factors:
                pos_encoding = positional_encoding(positions, base=1/2, d=d)
                local_min = np.min(pos_encoding)
                local_max = np.max(pos_encoding)
                global_min = min(global_min, local_min)
                global_max = max(global_max, local_max)

            for i, d in enumerate(d_factors, 1):
                pos_encoding = positional_encoding(positions, base=1/2, d=d)
                plt.subplot(2, 2, i)
                plt.imshow(pos_encoding.T, aspect='auto', cmap='viridis', origin='lower', vmin=global_min, vmax=global_max)
                plt.title(f'Positional Encodings (D: {d})')
                plt.xlabel('Position')
                plt.ylabel('Depth')
                plt.yticks(np.arange(0, 4), ['0', '1', '2', '3'])
                plt.colorbar()

            plt.tight_layout()
            plt.show()



def plot_embeddings(labels, coordinates, integer=False):
    plt.figure(figsize=(20, 4 if coordinates.shape[0] == 1 else 10))
    
    if coordinates.shape[0] == 1:  # Handle 1D embeddings
        # Extend to 2D for compatibility with scatter plot
        coordinates = np.vstack([coordinates, np.zeros_like(coordinates)])
        plt.ylim(-0.1, 0.1)
        plt.xlabel('Embedding Dim 1')
        plt.title('1D Embeddings')
        
        if integer:  # Set x-axis to show only integers
            x_ticks = np.arange(int(coordinates[0].min()), int(coordinates[0].max()) + 1)
            plt.xticks(x_ticks)
    else:  # Handle 2D embeddings
        plt.xlabel('Embedding Dim 1')
        plt.ylabel('Embedding Dim 2')
        plt.title('2D Embeddings')

    # Scatter plot and label annotations
    plt.scatter(coordinates[0], coordinates[1], marker='o', color='blue')
    for i, label in enumerate(labels):
        plt.text(
            coordinates[0, i], coordinates[1, i] + 0.01, f' {label}',
            verticalalignment='bottom', horizontalalignment='center', rotation='vertical'
        )

    plt.show()


def create_embeddings(dimensions, token_id=False):
    root_path = os.path.dirname(os.path.abspath(os.getcwd()))
    file_path = os.path.join(root_path, 'datasets', 'transformerDatasets', 'dummyDatasets', 'WordSimilarities')
    df = pd.read_csv(file_path, header=None)
    labels = pd.unique(df[[0, 1]].values.ravel('K'))

    item_to_index = {item: i for i, item in enumerate(labels)}

    num_items = len(labels)
    similarity_matrix = np.zeros((num_items, num_items))

    if token_id:
        return labels, np.arange(0, num_items)[None, :]

    for _, row in df.iterrows():
        idx1 = item_to_index[row[0]]
        idx2 = item_to_index[row[1]]
        similarity = row[2]
        similarity_matrix[idx1][idx2] = similarity
        similarity_matrix[idx2][idx1] = similarity

    _, embeddings = np.linalg.eig(similarity_matrix)

    return labels, embeddings[:, :dimensions].T


def get_measurement_data():
    x = np.linspace(0, 10, 500)
    noise = np.random.rand(500)
    a = np.random.rand(10)
    y = np.array([10 * np.sin(x), np.cos(x), np.sin(2 * x), np.cos(2 * x), np.sin(3 * x), np.cos(3 * x), np.sin(4 * x),
                  np.cos(4 * x), np.sin(5 * x), np.cos(5 * x)]).T @ a
    y_n = y + 3 * noise
    return {'time': x, 'data': y, 'data_noise': y_n}


def similarity(x1, x2):
    sigma = 0.05
    return np.exp(- ((x1 - x2) ** 2 / sigma ** 2))


def filter_data(measurement_data):
    w = similarity(measurement_data['time'][:, None], measurement_data['time'][None, :])

    w = w / np.sum(w, axis=0)

    measurement_data['data_filtered'] = w @ measurement_data['data_noise']
    return measurement_data


def load_word2vec():
    root_path = os.getcwd()
    file_path = os.path.join(root_path, 'models', 'pretrainedModels', 'w2v50.pkl')
    with open(file_path, 'rb') as file:
        word2vec = pickle.load(file)

    return word2vec


def embedd_sentence(sentence):
    root_path = os.getcwd()
    file_path = os.path.join(root_path, 'models', 'pretrainedModels', 'w2v50.pkl')
    with open(file_path, 'rb') as file:
        word2vec = pickle.load(file)

    embeddings = []
    words = []
    sentence = re.findall(r"[\w']+|[.,!?;]", sentence)
    for word in sentence:
        words.append(word.lower())
        embeddings.append(word2vec[word.lower()])

    embeddings = np.array(embeddings)

    return embeddings, words


def softmax(inputs: np.ndarray):
    inputs = np.exp(inputs)
    inputs /= np.sum(inputs, axis=-1, keepdims=True)
    return inputs


def plot_attention_scores(scores, words=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(scores, cmap='viridis', interpolation='nearest')  # Use 'viridis' colormap for clarity
    plt.colorbar()

    if words is not None:
        plt.xticks(np.arange(len(words)), words, rotation=90)
        plt.yticks(np.arange(len(words)), words)

        plt.xlabel('Words')
        plt.ylabel('Words')
        plt.title('Attention Scores')

    plt.show()


def get_dummy_embeddings():
    dummy_embedding = np.random.randint(1, 4, size=(4, 4))*0.5
    return dummy_embedding


def plot_boolean_masks(causal_masks, enc_masks):
    num_samples = len(causal_masks)
    seq_enc = enc_masks.shape[-1]
    enc_masks = enc_masks.expand(num_samples, seq_enc, seq_enc)

    plt.figure(figsize=(15, 5))

    for i in range(num_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(causal_masks[i], cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Attn Mask {i}')
        plt.axis('off')

        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(enc_masks[i], cmap='viridis', vmin=0, vmax=1)
        plt.title(f'Encoder Mask {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def visualize_scores(score_records, hparams):
    pass

def get_key(x):
    return x

def get_query(x):
    return x

def get_value(x):
    return x
