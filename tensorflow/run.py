
import tensorflow as tf

from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Permute, TimeDistributed, LSTM, Lambda, Dense
import matplotlib.pyplot as plt


def createModel(width: int):
    black, white = (0, 255) # TODO image preprocessing
    contrast = (white - black) / 2
    offset = black / contrast - 1

    model = tf.keras.Sequential([
        Rescaling(scale=1./contrast, offset=offset, input_shape=(36, width, 1)),
        Conv2D(16, 3, activation='tanh', padding="valid"),
        MaxPooling2D(3),
        Permute((2, 1, 3)),
        TimeDistributed(LSTM(48)),
        LSTM(96, return_sequences=True),
        LSTM(96, return_sequences=True, go_backwards=True),
        Lambda(lambda x: tf.reverse(x, [1])), # timestep reversed with backwards
        LSTM(192, return_sequences=True),
        TimeDistributed(Dense(111, activation='softmax'))
    ])

    from coeffs import conv, lstm_sum, lstm1, lstm2, lstm3, final_full

    model.get_layer(name="conv2d").set_weights([
        tf.reshape(conv[0], (3, 3, 1, 16)),
        tf.reshape(conv[1], (16))
    ])

    model.get_layer('time_distributed').set_weights([
        tf.reshape(lstm_sum[0], (16, 192)),
        tf.reshape(lstm_sum[1], (48, 192)),
        tf.reshape(lstm_sum[2], (192))
    ])

    model.get_layer('lstm_1').set_weights([
        tf.reshape(lstm1[0], (48, 384)),
        tf.reshape(lstm1[1], (96, 384)),
        tf.reshape(lstm1[2], (384))
    ])

    model.get_layer('lstm_2').set_weights([
        tf.reshape(lstm2[0], (96, 384)),
        tf.reshape(lstm2[1], (96, 384)),
        tf.reshape(lstm2[2], (384))
    ])

    model.get_layer('lstm_3').set_weights([
        tf.reshape(lstm3[0], (96, 768)),
        tf.reshape(lstm3[1], (192, 768)),
        tf.reshape(lstm3[2], (768))
    ])

    model.get_layer('time_distributed_1').set_weights([
        tf.reshape(final_full[0], (192, 111)),
        tf.reshape(final_full[1], (111)),
        ])

    model.summary()
    return model


def display_simple(img):
    plt.figure()
    plt.imshow(tf.squeeze(img), cmap='gray')
    plt.axis("off")
    plt.show()

def display_convs(result):
    components = tf.split(result, 16, axis=3)
    plt.figure(figsize=(200,200))
    for (i, component) in enumerate(components):
        tf.keras.utils.save_img("conv_2_%s.png" % i, tf.reshape(component, (228, 12, 1)))
        plt.subplot(2,8,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(tf.squeeze(component), cmap='gray')
        plt.xlabel(str(i))
    plt.show()

# TODO generate charset from data
charset = ['NULL', 'C', 'H', 'E', 'S', '-', 'R', 'I', 'K', 'N', 'G', 'B', '8', '5', 'F', ',', '(', '/', 'L', 'T', ')', 'O', 'Y', '.',
           'D', 'A', 'M', 'U', 'P', '[', ']', '9', '7', '0', '1', '4', '2', 'W', '3', '<', '>', '"', 'V', 'X', "'", '~', '!', 'J', 'Q',
           'Z', '+', '@', '&', '’', '=', '_', '€', '™', '“', '|', '?', ':', '6', '{', '}', '$', ';', '\\', '—', '”', '*', '#', '»', '®',
           '%', '£', '«', '°', '©', '§', '¥', '¢', '‘', 'i', 'n', 'c', 'u', 'l', 'a', 't', 'e', 'm', 'o', 's', 'g', 'h', 'b', 'z', 'v', 'q',
           'f', 'r', 'w', 'p', 'd', 'k', 'x', 'y', 'j', 'é', '|Broken|0|1']

# input is a preprocessed line
filename="../data/line.png"
img = tf.io.decode_png(tf.io.read_file(filename))
assert img.shape[0] == 36
width = img.shape[1]
model = createModel(width)
result = model(tf.reshape(img, (-1, 36, width, 1)))

# do a simple extraction of the characters (could plug into beam search)
r = tf.math.argmax(tf.reshape(result, (227, 111)), axis=1)
print(list(filter(lambda x: not x[0] == '|', map(lambda x: charset[x], r.numpy().tolist()))))
