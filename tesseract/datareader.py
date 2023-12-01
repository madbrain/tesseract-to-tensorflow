import struct
import os

# enum TessdataType
TESSDATA_LANG_CONFIG          = 0
TESSDATA_UNICHARSET           = 1
TESSDATA_AMBIGS               = 2
TESSDATA_INTTEMP              = 3
TESSDATA_PFFMTABLE            = 4
TESSDATA_NORMPROTO            = 5
TESSDATA_PUNC_DAWG            = 6
TESSDATA_SYSTEM_DAWG          = 7
TESSDATA_NUMBER_DAWG          = 8
TESSDATA_FREQ_DAWG            = 9
TESSDATA_FIXED_LENGTH_DAWGS   = 10  # deprecated
TESSDATA_CUBE_UNICHARSET      = 11  # deprecated
TESSDATA_CUBE_SYSTEM_DAWG     = 12  # deprecated
TESSDATA_SHAPE_TABLE          = 13
TESSDATA_BIGRAM_DAWG          = 14
TESSDATA_UNAMBIG_DAWG         = 15
TESSDATA_PARAMS_MODEL         = 16
TESSDATA_LSTM                 = 17
TESSDATA_LSTM_PUNC_DAWG       = 18
TESSDATA_LSTM_SYSTEM_DAWG     = 19
TESSDATA_LSTM_NUMBER_DAWG     = 20
TESSDATA_LSTM_UNICHARSET      = 21
TESSDATA_LSTM_RECODER         = 22
TESSDATA_VERSION              = 23


def read_components(file_name):
    file_size = os.stat(file_name).st_size
    
    f = open(file_name, "rb")
    num_entries = struct.unpack("I", f.read(4))[0]
    offsets = []
    for i in range(num_entries):
        offset = struct.unpack("q", f.read(8))[0]
        if offset >= 0:
            offsets.append((i, offset))
    
    entries = {}
    for (j, (i, offset)) in enumerate(offsets):
        if j < len(offsets)-1:
            size = offsets[j+1][1] - offset
        else:
            size = file_size - offset
        entries[i] = f.read(size)
    
    f.close()
    return entries

class DataStream:
    def __init__(self, data):
        self.data = data
        self.index = 0

    def read(self, fmt):
        size = struct.calcsize(fmt)
        result = struct.unpack_from(fmt, self.data, self.index)
        self.index += size
        return result

    def read_string(self):
        size = self.read("I")[0]
        result = self.data[self.index:self.index + size]
        self.index += size
        return result.decode('utf-8')
    
    def read_line(self):
        result = b''
        while True:
            v = self.data[self.index]
            self.index += 1
            if v == ord('\n'):
                break
            result += v.to_bytes(1, 'big')
        return result.decode('utf-8')

kTypeNames = [
    "Invalid",     "Input",
    "Convolve",    "Maxpool",
    "Parallel",    "Replicated",
    "ParBidiLSTM", "DepParUDLSTM",
    "Par2dLSTM",   "Series",
    "Reconfig",    "RTLReversed",
    "TTBReversed", "XYTranspose",
    "LSTM",        "SummLSTM",
    "Logistic",    "LinLogistic",
    "LinTanh",     "Tanh",
    "Relu",        "Linear",
    "Softmax",     "SoftmaxNoCTC",
    "LSTMSoftmax", "LSTMBinarySoftmax",
    "TensorFlow",
]

# enum NetworkFlags
# Network forward/backprop behavior.
NF_LAYER_SPECIFIC_LR = 64  # Separate learning rate for each layer.
NF_ADAM = 128              # Weight-specific learning rate.

class Plumbing:
    def __init__(self, name):
        self.name = name
        self.stack = []

    def deserialize(self, data):
        size = data.read("I")[0]
        print("[")
        for i in range(size):
            self.stack.append(read_network(data))
        if self.network_flags & NF_LAYER_SPECIFIC_LR != 0:
            self.learning_rates = read_vector(data, lambda: data.read("f")[0])
        print("]")


class Series(Plumbing):
    def __init__(self, name):
        super().__init__(name)

    def __str__(self):
        return "Series"


class Reversed(Plumbing):
    def __init__(self, name, ntype):
        super().__init__(name)
        self.type = ntype

    def __str__(self):
        return "Reversed[%d]" % (self.type)


# enum LossType
LT_NONE     = 0  # Undefined.
LT_CTC      = 1  # Softmax with standard CTC for training/decoding.
LT_SOFTMAX  = 2  # Outputs sum to 1 in fixed positions.
LT_LOGISTIC = 3  # Logistic outputs with independent values.

class StaticShape:
    def __init__(self, batch, height, width, depth, loss_type):
        self.batch = batch
        self.height = height
        self.width = width
        self.depth = depth
        self.loss_type = loss_type

    def __str__(self):
        return "%d:%d:%d:%d" % (self.batch, self.height, self.width, self.depth)

def read_shape(data):
    (batch, height, width, depth, loss_type) = data.read("iiiii")
    return StaticShape(batch, height, width, depth, loss_type)


class Input:
    def __init__(self, name, ni, no):
        self.name = name
        self.ni = ni
        self.no = no

    def deserialize(self, data):
        self.shape = read_shape(data)

    def __str__(self):
        return "Input([%s], %d, %d)" % (str(self.shape), self.ni, self.no)


class Convolve:
    def __init__(self, name, ni, half_x, half_y):
        self.name = name
        self.ni = ni
        self.no = ni * (2 * half_x + 1) * (2 * half_y + 1)
        self.half_x = half_x
        self.half_y = half_y

    def deserialize(self, data):
        (half_x, half_y) = data.read("ii")
        self.no = self.ni * (2 * half_x + 1) * (2 * half_y + 1)
        self.half_x = half_x
        self.half_y = half_y

    def __str__(self):
        return "Convolve([%dx%d], %d, %d)" % (self.half_x * 2 + 1, self.half_y * 2 + 1, self.ni, self.no)


class Maxpool:
    def __init__(self, name, ni, x_scale, y_scale):
        self.name = name
        self.ni = ni
        self.no = ni
        self.x_scale = x_scale
        self.y_scale = y_scale

    def deserialize(self, data):
        (x_scale, y_scale) = data.read("ii")
        self.x_scale = x_scale
        self.y_scale = y_scale
        # strange should have less outputs than inputs, doesn't seem to be used
        self.no = self.ni * self.x_scale * self.y_scale

    def __str__(self):
        return "Maxpool([%dx%d], %d, %d)" % (self.x_scale, self.y_scale, self.ni, self.no)


class FullyConnected:
    def __init__(self, name, ni, no, ntype):
        self.name = name
        self.ni = ni
        self.no = no
        self.type = ntype

    def deserialize(self, data):
        self.weights = read_weight_matrix(self.training, data)
        #print(self.weights.values.elements, self.weights.scales)

    def __str__(self):
        return "FullyConnected[%s](%d, %d, %s)" % (kTypeNames[self.type], self.ni, self.no, str(self.weights))

# enum WeightType
CI  = 0       # Cell Inputs.
GI  = 1       # Gate at the input.
GF1 = 2       # Forget gate at the memory (1-d or looking back 1 timestep).
GO  = 3       # Gate at the output.
GFS = 4       # Forget gate at the memory, looking back in the other dimension.
WT_COUNT = 5  # Number of WeightTypes.

class LSTM:
    def __init__(self, name, ni, ns, no, is_2d, ntype):
        self.name = name
        self.ni = ni
        self.no = no
        self.type = ntype
        self.ns = ns
        self.na = self.ni + self.ns
        self.nf = 0
        self.is2d = is_2d
        if is_2d:
            self.na += ns
        self.gate_weights = {}

    def deserialize(self, data):
        (na) = data.read("i")
        self.is_2d = False
        for w in range(WT_COUNT):
            if w == GFS and not self.is_2d:
                continue
            self.gate_weights[w] = read_weight_matrix(self.training, data)
            #print (w, self.gate_weights[w].values.elements, self.gate_weights[w].scales)
            if w == CI:
                self.ns = self.gate_weights[CI].num_outputs()
                self.is_2d = self.na - self.nf == self.ni + 2 * self.ns

    def __str__(self):
        return "LSTM[%s](%d, %d, %s, %d)" % (kTypeNames[self.type], self.ni, self.ns, str(self.gate_weights[CI]), self.no)


class Matrix:
    def __init__(self, width, height, empty, elements):
        self.width = width
        self.height = height
        self.empty = empty
        self.elements = elements

    def __str__(self):
        return "Matrix(%d,%d)" % (self.width, self.height)


def read_matrix(data, get_element):
    (size1, size2) = data.read("ii")
    empty = get_element()
    elements = [ get_element() for i in range(size1 * size2) ]
    return Matrix(size1, size2, empty, elements)

def read_vector(data, get_element):
    size = data.read("i")[0]
    return [ get_element() for i in range(size) ]


# Flag on mode to indicate that this weightmatrix uses int8_t.
kInt8Flag = 1
# Flag on mode to indicate that this weightmatrix uses adam.
kAdamFlag = 4
# Flag on mode to indicate that this weightmatrix uses double. Set
# independently of kInt8Flag as even in int mode the scales can
# be float or double.
kDoubleFlag = 128

class WeightMatrix:
    def __init__(self, values, scales):
        self.values = values
        self.scales = scales

    def num_outputs(self):
        return self.values.width

    def __str__(self):
        return "WeightMatrix(%d, %d+1)" % (self.values.width, self.values.height-1)

def read_weight_matrix(training, data) -> WeightMatrix:
    mode = data.read("=b")[0]
    int_mode = (mode & kInt8Flag) != 0
    use_adam = (mode & kAdamFlag) != 0
    if (mode & kDoubleFlag) == 0:
        raise Exception("HALT")
        if int_mode:
            wi = read_matrix(data, lambda: data.read("=b")[0])
            scales = read_vector(data, lambda: data.read("f")[0])
            return WeightMatrix(wi, scales)
        else:
            raise Exception("TODO continue DeSerializeOld(training, fp)")
    elif int_mode:
        wi = read_matrix(data, lambda: data.read("=b")[0])
        scales = read_vector(data, lambda: data.read("d")[0])
        return WeightMatrix(wi, scales)
    else:
        self.wf.DeSerialize(data)
        if training:
            raise Exception("TODO weight matrix training", int_mode, use_adam)
      #InitBackward();
      #if (!updates_.DeSerialize(fp)) return false;
      #if (use_adam_ && !dw_sq_sum_.DeSerialize(fp)) return false;


def read_network_type(data):
    network_type = data.read("=b")[0]
    if network_type == 0:
        name = data.read_string()
        for (i, n) in enumerate(kTypeNames):
            if name == n:
                return i
        raise Error()
    return network_type

def read_network(data):
    network_type = read_network_type(data)
    (training, needs_to_backprop, network_flags, ni, no, num_weights) = data.read("=bbiiii")
    name = data.read_string()
    if kTypeNames[network_type] == "Series":
        network = Series(name)
    elif kTypeNames[network_type] == "RTLReversed" or kTypeNames[network_type] == "XYTranspose":
        network = Reversed(name, network_type)
    elif kTypeNames[network_type] == "Input":
        network = Input(name, ni, no)
    elif kTypeNames[network_type] == "Convolve":
        network = Convolve(name, ni, 0, 0)
    elif kTypeNames[network_type] == "Maxpool":
        network = Maxpool(name, ni, 0, 0)
    elif kTypeNames[network_type] == "Tanh" or kTypeNames[network_type] == "Softmax":
        network = FullyConnected(name, ni, no, network_type)
    elif kTypeNames[network_type] == "LSTM" or kTypeNames[network_type] == "SummLSTM":
        network = LSTM(name, ni, no, no, False, network_type)
    else:
        print (kTypeNames[network_type])
        print (training, needs_to_backprop, network_flags, ni, no, num_weights)
        print(name)
        raise Error()
    network.training = training != 0
    network.needs_to_backprop = needs_to_backprop != 0
    network.network_flags = network_flags
    network.num_weight = num_weights
    network.deserialize(data)
    print("%s: %s" % (name, network))
    return network

kMaxCodeLen = 9

class RecodedCharID:
    def __init__(self, selfNormalized, codes):
        self.selfNormalized = selfNormalized
        self.codes = codes

    def length(self):
        return len(self.codes)

    def truncate(self, len):
        return RecodedCharID(self.selfNormalized, self.codes[0:len])
    
    def __eq__(self, other):
        return self.selfNormalized == other.selfNormalized and self.codes == other.codes

    def __hash__(self):
        return hash((self.selfNormalized, str(self.codes)))

    def __repr__(self) -> str:
        return "(%d, %s)" % (self.selfNormalized, str(self.codes))
    
def read_recoded_char(data):
    (selfNormalized, length) = data.read("=bi")
    codes = [ data.read("i")[0] for i in range(length) ]
    return RecodedCharID(selfNormalized, codes)

class UnicharCompress:
    def __init__(self):
        self.encoder = []

    def deSerialize(self, data):
        size = data.read("i")[0]
        self.encoder = [ read_recoded_char(data) for i in range(size) ]
        self.compute_code_range()
        self.setup_decoder()
    
    def compute_code_range(self):
        self.code_range = -1
        for ch in self.encoder:
            for c in ch.codes:
                self.code_range = max(self.code_range, c)
        self.code_range += 1

    def setup_decoder(self):
        self.decoder = {}
        self.final_codes = {}
        self.next_codes = {}
        self.is_valid_start = [ False for i in range(self.code_range) ]
        for (c, code) in enumerate(self.encoder):
            self.decoder[code] = c
            self.is_valid_start[code.codes[0]] = True
            l = code.length() - 1
            prefix = code.truncate(l)
            if prefix not in self.final_codes:
                code_list = [code.codes[l]]
                self.final_codes[prefix] = code_list
                while l > 0:
                    l -= 1
                    prefix = code.truncate(l)
                    if code_list not in self.next_codes:
                        code_list = [code.codes[l]]
                        self.next_codes[prefix] = code_list
                    else:
                        code_list = self.next_codes[prefix]
                        if code.codes[l] not in code_list:
                            code_list.append(code.codes[l])
                        break
            else:
                code_list = self.final_codes[prefix]
                if code.codes[l] not in code_list:
                    code_list.append(code.codes[l])

    def encode_unichar(self, unichar_id):
        if unichar_id < 0 or unichar_id >= len(self.encoder):
            return 0
        return self.encoder[unichar_id]
    
    def decodeUnichar(self, code):
        len = code.length()
        if len <= 0 or len > kMaxCodeLen:
            return INVALID_UNICHAR_ID
        if code not in self.decoder:
            return INVALID_UNICHAR_ID
        return self.decoder[code]


# enum TrainingFlags
TF_INT_MODE = 1
TF_COMPRESS_UNICHARSET = 64

# SpecialUnicharCodes
UNICHAR_SPACE  = 0
UNICHAR_JOINED = 1
UNICHAR_BROKEN = 2
INVALID_UNICHAR_ID = -1


class Recognizer:
    def __init__(self):
        self.recoder = UnicharCompress()
    
    def is_recoding(self):
        return (self.training_flags & TF_COMPRESS_UNICHARSET) != 0
    
    def read(self, components):
        data = DataStream(components[TESSDATA_LSTM])
        self.network = read_network(data)

        include_charsets = TESSDATA_LSTM_RECODER not in components.keys() or TESSDATA_LSTM_UNICHARSET not in components.keys()
        if include_charsets:
            self.charset = self.read_charsets(data, False)

        self.networkStr = data.read_string()
        (self.training_flags, training_iteration, sample_iteration, self.null_char, adam_beta, learning_rate, momentum) = data.read("iiiifff")
        if include_charsets:
            self.load_recoder(data)
        else:
            self.charset = self.read_charsets(DataStream(components[TESSDATA_LSTM_UNICHARSET]), False)
            self.load_recoder(DataStream(components[TESSDATA_LSTM_RECODER]))

    def read_charsets(self, data: DataStream, skip_fragments):
        count = int(data.read_line())
        charset = []
        for i in range(count):
            l = data.read_line().split(" ")
            charset.append(l[0])
        return charset

    def load_recoder(self, data: DataStream):
        if self.is_recoding():
            if self.recoder.deSerialize(data):
                return False
            code = self.recoder.encode_unichar(UNICHAR_SPACE);
            if code.codes[0] != UNICHAR_SPACE:
                raise Exception("Space was garbled in recoding!!");
        else:
            raise Exception("TODO")
            recoder_.SetupPassThrough(GetUnicharset());
            training_flags_ |= TF_COMPRESS_UNICHARSET;

# file_name = "/usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata"
# components = read_components(file_name)

# recognizer = Recognizer()
# recognizer.read(components)