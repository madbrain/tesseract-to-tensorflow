
import math

from tesseract.imageutils import save_to_image, save_to_image_line_contrast, get_component, transpose

class WeightMatrix:
    def __init__(self, values, scales):
        vectorSize = len(values) // len(scales)
        self.values = [ [ values[y*vectorSize + x] for x in range(vectorSize) ] for y in range(len(scales)) ]
        self.scales = scales
        self.shape = (len(scales), vectorSize)

    def dot(self, vector):
        result = []
        for (i, scale) in enumerate(self.scales):
            sum = 0
            for (j, v) in enumerate(self.values[i]):
                if j < len(vector):
                    sum += vector[j] * v
                elif j == len(vector):
                    result.append((sum + v) * scale)
                else:
                    raise Exception("wrong size")
        return result
    
class Convolution:
    def __init__(self, halfSize, weights):
        self.halfSize = halfSize
        self.weights = weights

    def process(self, inputData):
        height = len(inputData)
        width = len(inputData[0])
        result = []
        for y in range(len(inputData)):
            row = []
            for x in range(len(inputData[y])):
                vector = []
                for xx in range(x - self.halfSize, x + self.halfSize + 1):
                    for yy in range(y - self.halfSize, y + self.halfSize + 1):
                        if xx < 0 or yy < 0 or yy >= height or xx >= width:
                            vector.append(1.0) # WHITE COLOR
                        else:
                            vector.append(inputData[yy][xx])
                value = tanh(self.weights.dot(vector))
                row.append(value)
            result.append(row)
        return result
    
class Maxpool:
    def __init__(self, size):
        self.size = size

    def process(self, inputData):
        height = len(inputData)
        width = len(inputData[0])
        components = len(inputData[0][0])
        result = []
        for y in range(0, len(inputData), self.size):
            row = []
            for x in range(0, len(inputData[y]), self.size):
                maxes = [ None for i in range(components) ]
                for xx in range(self.size):
                    for yy in range(self.size):
                        xxx = x + xx
                        yyy = y + yy
                        if yyy < height and xxx < width:
                            for i in range(components):
                                if maxes[i] is None:
                                    maxes[i] = inputData[yyy][xxx][i]
                                else:
                                    maxes[i] = max(maxes[i], inputData[yyy][xxx][i])
                row.append(maxes)
            result.append(row)
        return result

def make_weight_matrix(obj):
    return WeightMatrix(obj.values.elements, obj.scales)

class TransposeIterator:
    def __init__(self, data):
        self.data = data
        self.index = (0, 0)
        self.result = []

    def next(self):
        (i, j) = self.index
        if i < len(self.data) and j < len(self.data[0]):
            v = self.data[i][j]
            i += 1
            if i >= len(self.data):
                i = 0
                j += 1
            self.index = (i, j)
            return v
        return None
    
    def isRowEnd(self):
        (i, j) = self.index
        return i == 0
    
    def push(self, value):
        self.result.append(value)

    def finish(self):
        return self.result
    
class NormalIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0
        self.result = []

    def next(self):
        if self.index < len(self.data):
            v = self.data[self.index]
            self.index += 1
            return v
        return None
    
    def isRowEnd(self):
        return self.index >= len(self.data)

    def push(self, value):
        self.result.append(value)

    def finish(self):
        return self.result
    
class ReversedIterator:
    def __init__(self, data):
        self.data = data
        self.index = 0
        self.result = []

    def next(self):
        if self.index < len(self.data):
            v = self.data[len(self.data) - 1 - self.index]
            self.index += 1
            return v
        return None
    
    def isRowEnd(self):
        return self.index >= len(self.data)

    def push(self, value):
        self.result.insert(0, value)

    def finish(self):
        return self.result
    
def softmax(vector):
    s = sum(map(lambda x: math.exp(x), vector))
    return [ math.exp(x) / s for x in vector ]
    
def sigmoid(vector):
    return [ 1 / (1 + math.exp(-x)) for x in vector ]

def tanh(vector):
    return [ math.tanh(x) for x in vector ]

def comp_mult(a, b):
    return [ x * y for (x, y) in zip(a, b) ]

def add_clip(a, b):
    def clip(x):
        return min(max(x, -100), 100)
    return [ clip(x + y) for (x, y) in zip(a, b) ]

class LSTM:
    def __init__(self, ci, gi, gf1, go, summarizing = False):
        self.ci = ci
        self.gi = gi
        self.gf1 = gf1
        self.go = go
        (i, j) = self.ci.shape
        self.stateSize = i
        self.summarizing = summarizing

    def process(self, it):
        state = [ 0.0 for i in range(self.stateSize) ]
        outputs = [ 0.0 for i in range(self.stateSize) ]
        while True:
            v = it.next()
            if v is None:
                break
            s = v + outputs
            state = add_clip(
                comp_mult(state, sigmoid(self.gf1.dot(s))),
                comp_mult(tanh(self.ci.dot(s)), sigmoid(self.gi.dot(s))))
            outputs = comp_mult(tanh(state), sigmoid(self.go.dot(s)))
            if not self.summarizing or it.isRowEnd():
                it.push(outputs)
            if it.isRowEnd():
                state = [ 0.0 for i in range(self.stateSize) ]
                outputs = [ 0.0 for i in range(self.stateSize) ]
        return it.finish()
    

class FullyConnected:
    def __init__(self, weights):
        self.weights: WeightMatrix = weights

    def process(self, inputData):
        return [ softmax(self.weights.dot(s)) for s in inputData ]


def apply_network(network, inputData):
    print("> Conv")
    conv = Convolution(1, make_weight_matrix(network.stack[1].stack[1].weights))
    convResult = conv.process(inputData)

    print("> MaxPool")
    maxpool = Maxpool(3)
    convResult = maxpool.process(convResult)

    for i in range(16):
        save_to_image("conv_%d" % i, get_component(convResult, i))

    print("> Summurazing LSTM")
    summarizingLSTM = LSTM(
        make_weight_matrix(network.stack[3].stack[0].gate_weights[0]),
        make_weight_matrix(network.stack[3].stack[0].gate_weights[1]),
        make_weight_matrix(network.stack[3].stack[0].gate_weights[2]),
        make_weight_matrix(network.stack[3].stack[0].gate_weights[3]),
        True
    )
    r = summarizingLSTM.process(TransposeIterator(convResult))
    save_to_image("sum_lstm_n", transpose(r))
    save_to_image_line_contrast("sum_lstm", transpose(r))

    print("> N1 LSTM")
    n1LSTM = LSTM(
        make_weight_matrix(network.stack[4].gate_weights[0]),
        make_weight_matrix(network.stack[4].gate_weights[1]),
        make_weight_matrix(network.stack[4].gate_weights[2]),
        make_weight_matrix(network.stack[4].gate_weights[3]),
        False
    )
    r = n1LSTM.process(NormalIterator(r))
    save_to_image("n1_lstm_n", transpose(r))
    save_to_image_line_contrast("n1_lstm", transpose(r))

    print("> N2 LSTM")
    n2LSTM = LSTM(
        make_weight_matrix(network.stack[5].stack[0].gate_weights[0]),
        make_weight_matrix(network.stack[5].stack[0].gate_weights[1]),
        make_weight_matrix(network.stack[5].stack[0].gate_weights[2]),
        make_weight_matrix(network.stack[5].stack[0].gate_weights[3]),
        False
    )
    r = n2LSTM.process(ReversedIterator(r))
    save_to_image("n2_lstm_n", transpose(r))
    save_to_image_line_contrast("n2_lstm", transpose(r))

    print("> N3 LSTM")
    n3LSTM = LSTM(
        make_weight_matrix(network.stack[6].gate_weights[0]),
        make_weight_matrix(network.stack[6].gate_weights[1]),
        make_weight_matrix(network.stack[6].gate_weights[2]),
        make_weight_matrix(network.stack[6].gate_weights[3]),
        False
    )
    r = n3LSTM.process(NormalIterator(r))
    save_to_image("n3_lstm_n", transpose(r))
    save_to_image_line_contrast("n3_lstm", transpose(r))

    print("> Final Full")
    final = FullyConnected(make_weight_matrix(network.stack[7].weights))
    r = final.process(r)
    save_to_image_line_contrast("final", transpose(r))

    return r