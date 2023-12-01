
import math

from tesseract.datareader import read_components, Recognizer

def to_conv_matrix(m):
    r = []
    s = int(math.sqrt(len(m.values[0])-1))
    for y in range(s):
        for x in range(s):
            for i in range(len(m.scales)):
                r.append(m.values[i][x*s+y] * m.scales[i])
    bias = [ m.values[i][-1]*m.scales[i] for i in range(len(m.scales)) ]
    return [ r, bias ]

def to_full_matrix(m):
    r = []
    for x in range(len(m.values[0])-1):
        for i in range(len(m.scales)):
            r.append(m.values[i][x] * m.scales[i])
    bias = [ m.values[i][-1]*m.scales[i] for i in range(len(m.scales)) ]
    return [ r, bias ]

def to_lstm_matrixes(ci, gi, gf1, go, isize):
    def conv(m):
        return [ [ x * s for x in m.values[i] ] for (i, s) in enumerate(m.scales) ]
    s = []
    for m in [ conv(gi), conv(gf1), conv(ci), conv(go) ]:
        for l in m:
            s.append(l)
    t = []
    for x in range(len(s[0])):
        l = []
        for y in range(len(s)):
            l.append(s[y][x])
        t.append(l)
    kernel = t[0:isize]
    recurrent_kernel = t[isize:len(t)-1]
    bias = t[-1]
    return [ kernel, recurrent_kernel, bias ]


class WeightMatrix:
    def __init__(self, values, scales):
        vectorSize = len(values) // len(scales)
        self.values = [ [ values[y*vectorSize + x] for x in range(vectorSize) ] for y in range(len(scales)) ]
        self.scales = scales
        self.shape = (len(scales), vectorSize)

def make_weight_matrix(obj):
    return WeightMatrix(obj.values.elements, obj.scales)


components = read_components("/usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata")
recognizer = Recognizer()
recognizer.read(components)

network = recognizer.network

f = open("tensorflow/coeffs.py", "w")

f.write("conv = %s;\n" % repr(to_conv_matrix(make_weight_matrix(network.stack[1].stack[1].weights))))

f.write("lstm_sum = %s;\n" % repr(to_lstm_matrixes(
        make_weight_matrix(network.stack[3].stack[0].gate_weights[0]),
        make_weight_matrix(network.stack[3].stack[0].gate_weights[1]),
        make_weight_matrix(network.stack[3].stack[0].gate_weights[2]),
        make_weight_matrix(network.stack[3].stack[0].gate_weights[3]),
        16)))

f.write("lstm1 = %s;\n" % repr(to_lstm_matrixes(
        make_weight_matrix(network.stack[4].gate_weights[0]),
        make_weight_matrix(network.stack[4].gate_weights[1]),
        make_weight_matrix(network.stack[4].gate_weights[2]),
        make_weight_matrix(network.stack[4].gate_weights[3]),
        48)))

f.write("lstm2 = %s;\n" % repr(to_lstm_matrixes(
        make_weight_matrix(network.stack[5].stack[0].gate_weights[0]),
        make_weight_matrix(network.stack[5].stack[0].gate_weights[1]),
        make_weight_matrix(network.stack[5].stack[0].gate_weights[2]),
        make_weight_matrix(network.stack[5].stack[0].gate_weights[3]),
        96)))

f.write("lstm3 = %s;\n" % repr(to_lstm_matrixes(
        make_weight_matrix(network.stack[6].gate_weights[0]),
        make_weight_matrix(network.stack[6].gate_weights[1]),
        make_weight_matrix(network.stack[6].gate_weights[2]),
        make_weight_matrix(network.stack[6].gate_weights[3]),
        96)))

f.write("final_full = %s;\n" % repr(to_full_matrix(make_weight_matrix(network.stack[7].weights))))

f.close()
