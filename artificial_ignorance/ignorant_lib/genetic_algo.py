import numpy as np
import pandas as p
import os
import datetime
import random
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import csv

print(f"Tensorflow version: {tf.__version__}")

# The following code will pin learning to the CPU only.
# First, enumerate all GPU devices:

#physical_devices = tf.config.list_physical_devices('GPU')

# Using the list of GPUs, if there are any, disable the first one by changing its visibility:
#try:
#  tf.config.set_visible_devices(physical_devices[1:], 'GPU')
#  logical_devices = tf.config.list_logical_devices('GPU')
#  assert len(logical_devices) == len(physical_devices) - 1
#except:
#  pass

"""
In this code we implement a genetic algorithmic approach toward a minimization problem.

    * Each of the top five previous models
    * Each of these will have three versions with minor (different) random fiddling
    * Remaining models will be generated randomly - For the first generation this would be all of them.
    
After each run, the top performing models are compared to the top 5 models overall.  The top five overall are retained.
All except for the top five in this just concluded generation are discarded.

Generations will continue until we reach a predefined accuracy, loss, or total number of generations
"""
GENERATION_SIZE = 10
MAX_GENERATIONS = 3
DESIRED_ACCURACY = 1.01 # Impossible, guaranteeing max_generations
EPOCHS = 3
ACTIVATIONS = ["relu", "tanh", "sigmoid"]
OPTIMIZERS = ["adam", "rmsprop", "adamax"]
MAX_NEURONS = 32
BUILD_CNN = True
MAX_FILTERS = 32
MIN_FILTERS = 4
MIN_KERNEL_SIZE = 2
MAX_KERNEL_SIZE = 6
MAX_LAYERS = 8
BATCH_SIZE = 64

def create_model(cnn_layers=[], dense_layers=[], activations=['relu'], optimizer='adam'):
    if len(dense_layers)+len(cnn_layers) != len(activations):
        print("The number of activations must match the total number of layers!")
        return
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=X.shape[1:]))
    if (BUILD_CNN): # We want convolutional layers
        for i, layer in enumerate(cnn_layers):
            model.add(layers.Conv2D(filters=layer[0], kernel_size=layer[1], strides=layer[2], 
                                    activation=activations[i], padding='same'))
        model.add(layers.Flatten())
    for i,layer in enumerate(dense_layers):
        model.add(layers.Dense(layer, activation=activations[i]))

    # We always need an output layer
    # Single neuron,sigmoid for binary
    model.add(layers.Dense(1, activation='sigmoid'))
    # Multi-class dense output layer with softmax
    #model.add(layers.Dense(len(uniqueLabels), activation='softmax'))

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def generate_random_model():
    """Generate a completely random model"""
    num_layers = random.randrange(2,MAX_LAYERS+1) # A single layer would never be ok
    layers = []
    cnn_layers = []
    activations = []
    if(BUILD_CNN):
        num_cnn_layers = random.randrange(0, num_layers+1)
        for i in range(0,num_layers):
            if(num_cnn_layers > 0):
                cnn_layers.append((
                    random.randrange(MIN_FILTERS,MAX_FILTERS+1), 
                    random.randrange(MIN_KERNEL_SIZE, MAX_KERNEL_SIZE+1),
                    1))
                activations.append(ACTIVATIONS[random.randrange(0, len(ACTIVATIONS))])
                num_cnn_layers -= 1
            else:
                layers.append(random.randrange(1,MAX_NEURONS))
                activations.append(ACTIVATIONS[random.randrange(0,len(ACTIVATIONS))])
    return ({"cnn_layers":cnn_layers, "layers":layers, "origin":"random", "born":generation_number, "age":0, "activations":activations, "optimizer":OPTIMIZERS[random.randrange(0,len(OPTIMIZERS))], "batch":BATCH_SIZE, "eval":(0,0)})

def run_generation(models):
    """Process a generation"""
    print("Generation Running")
    for i, raw_model in enumerate(models):
        print(f'Model {i} of {len(models)}')
        #print_model(raw_model)
        # Advance the age so that if this model survives, we know how many generations it has survived through.
        raw_model["age"] = raw_model["age"] + 1
        #logdir = os.path.join(LOG_PARENT, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        #tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        model = create_model(cnn_layers=raw_model['cnn_layers'], dense_layers=raw_model["layers"], 
                             activations=raw_model["activations"], optimizer=raw_model["optimizer"])
        training_history = model.fit(X, Y, validation_split=0.1, epochs=EPOCHS, verbose=0, shuffle=True,  
                                     batch_size=raw_model["batch"]) 
        raw_model["eval"] = model.evaluate(X_test, Y_test, verbose=0)

def breed_average(mother, father):
    cnn_layers = mother['cnn_layers']
    layers = []
    activations = mother['activations'][:len(cnn_layers)] if len(cnn_layers) else []
    # How many layers will we have?  Let's take the average of mom and dad.
    num_layers = (len(mother["layers"]) + len(father["layers"]))//2
    for i in range(0,num_layers):
        # It's possible that mom and dad have different numbers of layers.
        # If one model runs out of layers, default to using the data from the other model.
        # This is still performed as an average, but it all works out since (2x)/2 = x
        try:
            mother_neurons = mother["layers"][i]
        except:
            mother_neurons = father["layers"][i]
        try:
            father_neurons = father["layers"][i]
        except:
            father_neurons = mother["layers"][i]
        layers.append((father_neurons + mother_neurons)//2)
        activations.append(ACTIVATIONS[random.randrange(0,len(ACTIVATIONS))])
        # Set the "origin" of this model to be "average child"
    return({"cnn_layers":cnn_layers, "layers":layers, "eval":(0,0), "origin":"average child", "born":generation_number, "age":0, "activations":activations, "optimizer":OPTIMIZERS[random.randrange(0,len(OPTIMIZERS))], "batch":BATCH_SIZE})


def flatten_list(data):
    # iterating over the data
    def flat_the_list(data):
        for element in data:
            # checking for list
            if type(element) == list:
                # calling the same function with current element as new argument
                flat_the_list(element)
            else:
                flat_list.append(element)    
    flat_list = []
    flat_the_list(data)
    return flat_list

def breed_mutate(mother, father):
    cnn_layers = mother['cnn_layers']
    layers = []
    activations = mother['activations'][:len(cnn_layers)] if len(cnn_layers) else []
    num_layers = (len(mother["layers"]) + len(father["layers"]))//2
    for i in range(0,num_layers):
        # Work out the neurons and activations.
        # Use try...except to handle issues where one model has more layers than the other.
        try:
            mother_neurons = mother["layers"][i]
            mother_activation = mother["activations"][i]
        except:
            mother_neurons = father["layers"][i]
            mother_activation = father["activations"][i]
        try:
            father_neurons = father["layers"][i]
            father_activation = father["activations"][i]
        except:
            father_neurons = mother["layers"][i]
            father_activation = mother["activations"][i]
            
        # Keep either the father, the mother, or a combination of the two.
        which_to_keep = random.randrange(0,4)
        if(which_to_keep == 0):
            layers.append(father_neurons)
            activations.append(father_activation)
        if(which_to_keep == 1):
            layers.append(mother_neurons)
            activations.append(mother_activation)
        if(which_to_keep == 2):
            layers.append(mother_neurons)
            activations.append(father_activation)
        if(which_to_keep == 3):
            layers.append(father_neurons)
            activations.append(mother_activation)
    return({"cnn_layers":cnn_layers, "layers":layers, "eval":(0,0), "origin":"mutated child", "born":generation_number, "age":0, "activations":activations, "optimizer":OPTIMIZERS[random.randrange(0,len(OPTIMIZERS))], "batch":BATCH_SIZE})

def evolve_model(current_model):
    """Take the current model and return a list of three evolved models."""
    evolved = []
    evolved.append(current_model)
    cnn_layers = current_model['cnn_layers']
    activations = current_model['activations'][:len(cnn_layers)] if len(cnn_layers) else []
    cnn_activations = current_model['activations'][:len(cnn_layers)] if len(cnn_layers) else []
    
    layers = [layer*random.randrange(1,8) for layer in current_model["layers"]]
    layers = [layer if layer<MAX_NEURONS else MAX_NEURONS for layer in layers]
    if random.randrange(0,10)==1:
        layers.append(random.randrange(1,MAX_NEURONS))
    if random.randrange(0,10)==1 and len(layers) > 1:
        # 10% chance that we delete a layer
        layers.pop()
    activations = [ACTIVATIONS[random.randrange(0,len(ACTIVATIONS))] for layer in layers]
    evolved.append({"cnn_layers":cnn_layers, "layers":layers, "eval":(0,0), "origin":"mutation", "born":generation_number, "age":0, "activations":flatten_list([cnn_activations, activations]), "optimizer":OPTIMIZERS[random.randrange(0,len(OPTIMIZERS))], "batch":BATCH_SIZE})
    layers = [(layer//random.randrange(1,8))+1 for layer in current_model["layers"]]
    if random.randrange(0,10)==1:
        layers.append(random.randrange(1,MAX_NEURONS))
    if random.randrange(0,10)==1 and len(layers) > 1:
        # 10% chance that we delete a layer
        layers.pop()
    activations = [ACTIVATIONS[random.randrange(0,len(ACTIVATIONS))] for layer in layers]
    evolved.append({"cnn_layers":cnn_layers, "layers":layers, "eval":(0,0), "origin":"mutation", "born":generation_number, "age":0, "activations":flatten_list([cnn_activations, activations]), "optimizer":OPTIMIZERS[random.randrange(0,len(OPTIMIZERS))], "batch":BATCH_SIZE})
    return evolved

def compare_models(a,b):
    if a == b:
        # This is a false true because they are actually the same model.
        return True
    if len(a['layers']) != len(b['layers']):
        return False
    for i in range(0,len(a['layers'])):
        if a['layers'][i] != b['layers'][i] or a['activations'][i] != b['activations'][i]:
            return False
#    print(f"Duplicate models:")
#    print("-----------------------------")
#    print_model(a)
#    print_model(b)
#    print("-----------------------------")    
    return True

def prune_duplicates(models):
    results = [models[0]]
    for a in models:
        add_this = True
        for b in results:
            if compare_models(a,b) == True:
                add_this = False
        if add_this == True:
            results.append(a) 
    return results

def print_model(m):
    print(f"  Loss: {m['eval'][0]:4f}  Accuracy: {m['eval'][1]*100.0:2f}% Born: {m['born']}  Age: {m['age']} Origin: {m['origin']}")
    print(f"    CNN Layers: {len(m['cnn_layers'])} Layers: {len(m['layers'])}  Neurons: {m['layers']}  Batch: {m['batch']}")
    for i, cnn in enumerate(m['cnn_layers']):
        print(f"      CNN-{i}: {cnn[0]} {cnn[1]}x{cnn[1]} filters")  

    print(f"    Activations: {m['activations']}")

def better_model(a, b):
    (aloss, aaccuracy) = a.evaluate(X_test, Y_test)
    (bloss, baccuracy) = b.evaluate(X_test, Y_test)
    if aaccuracy > baccuracy:
        return a
    return b

def find_top_five(models):
    """Return a list of the top 5 scoring models"""
    return sorted(models, key=lambda a:a["eval"][0])[0:5]

generation_number = 0

def evolve():
    global generation_number
    print(f'Generating {GENERATION_SIZE} random initial models for the first generation.')
    generation = list()
    for i in range(GENERATION_SIZE):
        generation.append(generate_random_model())

    done = False
    best_overall = []
    while not done:
        print(f"Pruning any duplicate models... Starting with {len(generation)}")
        generation = prune_duplicates(generation)
        print(f"Ended with {len(generation)}")
        if(len(generation) < GENERATION_SIZE):
            additional_needed = GENERATION_SIZE - len(generation)
            print(f"Adding {additional_needed} random models.")
            for i in range(0,additional_needed):
                generation.append(generate_random_model())
        generation_number = generation_number + 1
        print(f"Beginning generation {generation_number}:")
        run_generation(generation)
        best_five = find_top_five(generation)
        print(f"Top five:")
        for m in best_five:     
            print_model(m)
        if best_overall:
            best_overall = find_top_five(best_five+best_overall)
            best_five = best_overall
            print(f"Best five overall:")
            for m in best_five:      
                print_model(m)
        else:
            best_overall = best_five
        generation = []
        print("Marrying best of breed...")
        for mother in best_five:
            for father in best_five:
                if(not mother == father):
                    generation.append(breed_average(mother,father))
                    generation.append(breed_mutate(mother,father))
        print("Evolving...")
        for model in best_five:
            generation.append(model)
            for evolved in evolve_model(model):
                generation.append(evolved)
        if generation_number >= MAX_GENERATIONS or best_overall[0]["eval"][1] > DESIRED_ACCURACY:
            print("Met stopping criteria!")
            done = True
    for m in best_five:
        print_model(m)
    print(f"Best overall model: {best_overall[0]}")

    import numpy as np
import os
import datetime
import re
import matplotlib.pyplot as plt

def plot_history(history):
    accuracy = history['accuracy']
    loss = history['loss']
    val_accuracy = history['val_accuracy']
    val_loss = history['val_loss']
    (figure, axes) = plt.subplots(1,2,figsize=(8,4), dpi=300)
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
    # We'll use a regular expression to find things that are not words or spaces.
    regex = re.compile("[^\w\s]")
    # Start with an empty list
    words = list()
    # Open the specified file
    with open(file_name, encoding='utf8', errors='ignore') as f:
        # Grab all of the lines
        text = f.readlines()
        # Set a flag to keep track of whether we have reached the body or not.
        finished_header = False
        # Iterate over the lines
        for line in text:
            # The last line in the headers is consistently the subject line.  If
            # we have not yet seen the subject then we are still parsing headers
            # and should ignore them.
            if finished_header:
                # If we have seen subject line, let's strip out things that aren't words
                # spaces, lowercase the line, and split it on whitespace.  Each word in
                # this list is then appended to the accumulating word list.
                for word in re.sub(regex, '', line.lower()).split():
                    words.append(word)
            # Check to see if the beginning of the line contains "subject:"
            # to determine if we have reached the end of the email header.
            elif line.lower() == "\n":
                # If we have, set the flag
                finished_header = True
    return words



all_words = dict()
for file in get_file_list("../data/Enron"):
    words = get_words(file)
    for word in words:

        if all_words.__contains__(word)==True:
            all_words[word] = all_words[word]+1
        else:
            all_words[word] = 1

def tokenize_email(text, words=100):
    """
    Accepts a list of words and a number of words from the dictionary to use.  Returns
    a list of word indices for each word.
    """
    word_array = [word_index[w] if word_index[w] < words else 0 for w in text]
    return word_array

def untokenize_email(indices):
    """
    Accepts a list of word indices.  Returns a list of words.
    """
    return [reverse_word_index[i] for i in indices]
    
            
total_words = len(all_words)            
print(f"Total unique words: {total_words}")
word_dictionary = sorted(all_words.items(), key=lambda key_value: key_value[1], reverse=True)
word_index = {k[0]:i+1 for i, k in enumerate(sorted(word_dictionary, key=lambda k_v: k_v[1], reverse=True))}
word_index["___"] = 0
reverse_word_index = {k:v for v, k in word_index.items()}

list_data = list()
for file in get_file_list("../data/Enron/ham"):
    list_data.append((get_words(file)))
    
x_ham_data = np.array(list_data, dtype=object)

list_data = list()  # Clear the list before we start adding spam
for file in get_file_list("../data/Enron/spam"):
    list_data.append((get_words(file)))
    
x_spam_data = np.array(list_data, dtype=object)


print(f"Loaded ham and spam.  Ham: {x_ham_data.shape} Spam: {x_spam_data.shape}")

sequence_length = 700

from tensorflow.keras.preprocessing.sequence import pad_sequences

def encode_text_Token(tokenizer, lines, length):
    # integer encode
    encoded = tokenizer.texts_to_sequences(lines)
    # pad encoded sequences
    padded = pad_sequences(encoded, maxlen=length, padding='post', truncating='post')
    return padded

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(np.append(x_ham_data, x_spam_data, axis=0))
voc_size =  len(tokenizer.word_index) + 1
print(f"Vocabulary size {voc_size}")
x_ham_encoded = encode_text_Token(tokenizer,x_ham_data, sequence_length)
x_spam_encoded = encode_text_Token(tokenizer, x_spam_data, sequence_length)

merged_data = np.append(x_ham_encoded, x_spam_encoded, axis=0)
labels = np.zeros(x_ham_encoded.shape[0])
labels = np.append(labels, np.ones(x_spam_encoded.shape[0]))
labels = labels.reshape(labels.shape[0],1)
all_data = np.append(labels, merged_data, axis=1)
all_data.shape

np.random.shuffle(all_data)
all_x = all_data[:, 1:]
all_y = all_data[:, 0]

x_train = all_x[:-1000]
y_train = all_y[:-1000]
x_test = all_x[-1000:]
y_test = all_y[-1000:]

x_train_reshape = x_train.reshape(-1,10,70,1)
x_test_reshape = x_test.reshape(-1, 10, 70, 1)

X = x_train_reshape
Y = y_train
X_test = x_test_reshape
Y_test = y_test

evolve()