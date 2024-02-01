import numpy as np
import matplotlib.pyplot as plt
import os
import sklearn


def plot_history(history):
    accuracy = history['accuracy']
    loss = history['loss']
    val_accuracy = history['val_accuracy']
    val_loss = history['val_loss']
    
    (figure, axes) = plt.subplots(1,2,figsize=(10,4), dpi=300)
    axes[0].plot(accuracy, label="Training")
    axes[0].plot(val_accuracy, label="Validation")
    axes[0].legend()
    axes[0].set_xlabel("Epoch")
    axes[0].grid()
    axes[0].set_title("Accuracy")

    axes[1].plot(loss, label="Training")
    axes[1].plot(val_loss, label="Validation")
    axes[1].legend()
    axes[1].set_xlabel("Epoch")
    axes[1].grid()
    axes[1].set_title("Loss")
    plt.show()

def sentiment(value):
    return "Positive" if value == 1 else "Negative"

def vectorize(data, dimensions):
    multihot = np.zeros((len(data), dimensions))
    for row, col in enumerate(data):
        multihot[row, col] = 1
    return multihot

def scatter3d(data, labels):
    clusters = len(np.unique(labels))
    colors=list("rgbcmyk")
    fig = plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    ax = fig.add_subplot(111, projection='3d')
    for cluster in range(clusters):
        ax.scatter(data[labels==cluster, 0], data[labels==cluster, 1], data[labels==cluster, 2], c=colors[cluster%len(colors)])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def get_file_list(starting_directory="../data/Enron/"):
    final_list = list()
    files = os.listdir(starting_directory)
    for file in files:
        file_name = os.path.join(starting_directory, file)
        if os.path.isdir(file_name):
            final_list = final_list + get_file_list(file_name)
        else:
            final_list.append(file_name)
    return final_list

def get_words(file_name):
    regex = re.compile("[^\w\s]")
    words = list()
    with open(file_name, encoding='utf8', errors='ignore') as f:
        text = f.readlines()
        finished_header = False
        for line in text:
            if finished_header:
                for word in re.sub(regex, '', line.lower()).split():
                    words.append(word)
            elif line.lower() == "\n":
                finished_header = True
    return words

def tokenize_email(text, word_index, words=100):
    # word_index = {k[0]:i+1 for i, k in enumerate(sorted(word_dictionary, key=lambda k_v: k_v[1], reverse=True))}
    # word_index["___"] = 0
    word_array = [word_index[w] if word_index[w] < words else 0 for w in text]
    return word_array

def untokenize_email(indices, reverse_word_index):
    # reverse_word_index = {k:v for v,k in word_index.items()}
    return [reverse_word_index[i] for i in indices]

def vectorize_sequence(word_index_array, dimension=WordsToWorkWith):
    results = np.zeros((len(word_index_array), dimension))
    for i, word in enumerate(word_index_array):
        results[i, word] = 1.0
    return results

def mse(y1, y2):
    if len(y1) != len(y2):
        print("bad lens")
        return
    return np.square(y1 - y2).mean()

