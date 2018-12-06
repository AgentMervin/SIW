# coding: utf-8

from distutils.version import LooseVersion
import warnings
import tensorflow as tf
import helper
import numpy as np
from tensorflow.contrib import seq2seq
import pickle


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    vocab_to_int = {w: i for i, w in enumerate(set(text))}
    int_to_vocab = {i: w for i, w in enumerate(set(text))}
    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    return {
        '.': '||Period||',
        ',': '||Comma||',
        '"': '||Quotation_Mark||',
        ';': '||Semicolon||',
        '!': '||Exclamation_mark||',
        '?': '||Question_mark||',
        '(': '||Left_Parentheses||',
        ')': '||Right_Parentheses||',
        '--': '||Dash||',
        "\n": '||Return||'
    }


def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    # TODO: Implement Function
    inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input')
    targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='targets')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    return inputs, targets, learning_rate


def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    # TODO: Implement Function
    keep_prob = 0.8
    layers = 3
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * layers)

    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')
    return cell, initial_state


def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    # TODO: Implement Function
    embeddings = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed = tf.nn.embedding_lookup(embeddings, input_data)
    return embed


def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    # TODO: Implement Function
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, 'final_state')
    return outputs, final_state


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    # TODO: Implement Function
    embed = get_embed(input_data, vocab_size, rnn_size)
    outputs, final_state = build_rnn(cell, embed)
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size, activation_fn=None)
    return logits, final_state


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    # TODO: Implement Function
    n_batches = len(int_text) // (batch_size * seq_length)
    result = []
    for i in range(n_batches):
        inputs = []
        targets = []
        for j in range(batch_size):
            idx = i * seq_length + j * seq_length
            inputs.append(int_text[idx:idx + seq_length])
            targets.append(int_text[idx + 1:idx + seq_length + 1])
        result.append([inputs, targets])
    return np.array(result)


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    # TODO: Implement Function
    return int_to_vocab[np.argmax(probabilities)]


def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # TODO: Implement Function
    inputs = loaded_graph.get_tensor_by_name('input:0')
    init_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probs = loaded_graph.get_tensor_by_name('probs:0')
    return inputs, init_state, final_state, probs


def generate(path, length, protagonist):
    corpus_raw=u""
    for data_dir in path:
        text = helper.load_data(data_dir)
        corpus_raw +=text

    token_dict = token_lookup()
    for token, replacement in token_dict.items():
          corpus_raw = corpus_raw.replace(token, ' {} '.format(replacement))
    corpus_raw = corpus_raw.lower()
    corpus_raw = corpus_raw.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(corpus_raw)
    corpus_int = [vocab_to_int[word] for word in corpus_raw]
    pickle.dump((corpus_int, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))


    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

    num_epochs = 100
    batch_size = 128
    rnn_size = 256
    embed_dim = 300
    seq_length = 25
    learning_rate = 0.01
    show_every_n_batches = 50
    save_dir = './save'


    train_graph = tf.Graph()
    with train_graph.as_default():
        #initialize input placeholders
        input_text, targets, lr = get_inputs()
        #Calculate text attributes
        vocab_size = len(int_to_vocab)
        input_data_shape = tf.shape(input_text)
        #set initial state
        cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
        logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)
        # Probabilities for generating words
        probs = tf.nn.softmax(logits, name='probs')

        # Loss function
        cost = seq2seq.sequence_loss(
            logits,
            targets,
            tf.ones([input_data_shape[0], input_data_shape[1]]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

        #tranin the network
        pickle.dump((seq_length, save_dir), open('params.p', 'wb'))
        batches = get_batches(corpus_int, batch_size, seq_length)

        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(num_epochs):
                state = sess.run(initial_state, {input_text: batches[0][0]})

                for batch_i, (x, y) in enumerate(batches):
                    feed = {
                        input_text: x,
                        targets: y,
                        initial_state: state,
                        lr: learning_rate}
                    train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

                    # Show every <show_every_n_batches> batches
                    if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                            epoch_i,
                            batch_i,
                            len(batches),
                            train_loss))

            saver = tf.train.Saver()
            saver.save(sess, save_dir)
            print('Model Trained and Saved')

    helper.save_params((seq_length, save_dir))

    corpus_int, vocab_to_int, int_to_vocab, token_dict = pickle.load(open('preprocess.p', mode='rb'))
    seq_length, save_dir = pickle.load(open('params.p', mode='rb'))
    get_length = length
    prime_word = protagonist

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Get Tensors from loaded model
        input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

        # Sentences generation setup
        gen_sentences = [prime_word + ':']
        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

        # Generate sentences
        for n in range(gen_length):
            # Dynamic Input
            dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])

            # Get Prediction
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, initial_state: prev_state})

            pred_word = pick_word(probabilities[dyn_seq_length - 1], int_to_vocab)

            gen_sentences.append(pred_word)

        # Remove tokens
        tv_script = ' '.join(gen_sentences)
        for key, token in token_dict.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            tv_script = tv_script.replace(' ' + token.lower(), key)
        tv_script = tv_script.replace('\n ', '\n')
        tv_script = tv_script.replace('( ', '(')
        return tv_script
