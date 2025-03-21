import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import re


def preproc(x):
    x = re.sub(r"[^a-zA-Z0-9\s]", "", x)
    x = x.lower()
    return x


def preproc_spanish(text):
    text = re.sub(r"[^a-zA-Z0-9áéíóúüñÁÉÍÓÚÜÑ\s]", "", text)
    text = text.lower()
    return text


df = pd.read_csv("spa.txt.csv")

df["English"] = df["English"].apply(preproc)
df["Translated"] = df["Translated"].apply(preproc_spanish)
eng_sentences = df["English"].values
spa_sentences = df["Translated"].values

# Tokenization
eng_tokenizer = Tokenizer(filters="")
spa_tokenizer = Tokenizer(filters="")
eng_tokenizer.fit_on_texts(eng_sentences)
spa_tokenizer.fit_on_texts(spa_sentences)

eng_vocab_size = len(eng_tokenizer.word_index) + 1
spa_vocab_size = len(spa_tokenizer.word_index) + 1

eng_sequences = eng_tokenizer.texts_to_sequences(eng_sentences)
spa_sequences = spa_tokenizer.texts_to_sequences(spa_sentences)

max_eng_len = max(len(seq) for seq in eng_sequences)
max_spa_len = max(len(seq) for seq in spa_sequences)

eng_padded = pad_sequences(eng_sequences, maxlen=max_eng_len, padding="post")
spa_padded = pad_sequences(spa_sequences, maxlen=max_spa_len, padding="post")
print("English Padded Shape:", eng_padded.shape)
print("Spanish Padded Shape:", spa_padded.shape)


# Define Encoder
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(enc_units, return_sequences=True, return_state=True)

    def call(self, x):
        x = self.embedding(x)
        output, state_h, state_c = self.lstm(x)
        return output, state_h, state_c


# Define Bahdanau Attention
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * values, axis=1)
        return context_vector, attention_weights


# Define Decoder
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(dec_units, return_sequences=True, return_state=True)
        self.fc = Dense(vocab_size, activation="softmax")
        self.attention = BahdanauAttention(dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden[0], enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state_h, state_c = self.lstm(x, initial_state=hidden)
        return self.fc(output), state_h, state_c


# Define Seq2Seq Model
class Seq2Seq(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs, training=False):
        inputs, targets = inputs
        enc_output, enc_hidden_h, enc_hidden_c = self.encoder(inputs)
        dec_hidden = [enc_hidden_h, enc_hidden_c]
        dec_input = tf.expand_dims(targets[:, 0], 1)  # Start with first word

        all_predictions = []
        for t in range(targets.shape[1]):
            predictions, dec_hidden_h, dec_hidden_c = self.decoder(
                dec_input, dec_hidden, enc_output
            )
            all_predictions.append(predictions)
            dec_hidden = [dec_hidden_h, dec_hidden_c]
            dec_input = tf.expand_dims(targets[:, t], 1)

        x = tf.concat(all_predictions, axis=1)
        return x


def masked_loss(y_true, y_pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss = loss_object(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_mean(loss)


# Set hyperparameters
embedding_dim = 128
units = 8
batch_size = 96

encoder = Encoder(eng_vocab_size, embedding_dim, units)
decoder = Decoder(spa_vocab_size, embedding_dim, units)

# Initialize model
seq2seq_model = Seq2Seq(encoder, decoder)

# Compile model
seq2seq_model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=masked_loss,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

# Prepare Dataset
dataset = tf.data.Dataset.from_tensor_slices(
    ((eng_padded, spa_padded[:, :-1]), spa_padded[:, 1:])
)
dataset = dataset.batch(batch_size).shuffle(len(eng_padded))

# Train Model
seq2seq_model.fit(dataset, epochs=50)

# Save Model
seq2seq_model.save("seq2seq_model.h5")
