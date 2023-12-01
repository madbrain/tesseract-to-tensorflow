
import math

from tesseract.datareader import RecodedCharID

TN_TOP2     = 0 # Winner or 2nd.
TN_TOPN     = 1 # Runner up in top-n, but not 1st or 2nd.
TN_ALSO_RAN = 2 # Not in the top-n.
TN_COUNT    = 3

# Scale factor to make certainty more comparable to Tesseract.
kCertaintyScale = 7.0
# Worst acceptable certainty for a dictionary word.
kWorstDictCertainty = -25.0

INVALID_UNICHAR_ID = -1
UNICHAR_SPACE  = 0
UNICHAR_JOINED = 1
UNICHAR_BROKEN = 2

# enum PermuterType
NO_PERM            = 0
PUNC_PERM          = 1
TOP_CHOICE_PERM    = 2
LOWER_CASE_PERM    = 3
UPPER_CASE_PERM    = 4
NGRAM_PERM         = 5
NUMBER_PERM        = 6
USER_PATTERN_PERM  = 7
SYSTEM_DAWG_PERM   = 8
DOC_DAWG_PERM      = 9
USER_DAWG_PERM     = 10
FREQ_DAWG_PERM     = 11
COMPOUND_PERM      = 12

# enum NodeContinuation
NC_ANYTHING = 0  # This node used just its own score, so anything can follow.
NC_ONLY_DUP = 1  # The current node combined another score with the score for
                 # itself, without a stand-alone duplicate before, so must be
                 # followed by a stand-alone duplicate.
NC_NO_DUP = 2    # The current node combined another score with the score for
                 # itself, after a stand-alone, so can only be followed by
                 # something other than a duplicate of the current node.

class BeamIndex:
    def __init__(self, is_dawg: bool, cont: int, length: int):
        self.is_dawg = is_dawg
        self.cont = cont
        self.length = length

    def __eq__(self, other: object) -> bool:
        return self.is_dawg == other.is_dawg and self.cont == other.cont and self.length == other.length
    
    def __hash__(self) -> int:
        return hash((self.is_dawg, self.cont, self.length))

    def __repr__(self) -> str:
        contStr = "ANY"
        if self.cont == NC_ONLY_DUP:
            contStr = "ONLY_DUP"
        elif self.cont == NC_NO_DUP:
            contStr = "NO_DUP"
        return "(%r, %s, %d)" % (self.is_dawg, contStr, self.length)

class RecodeNode:
    def __init__(self, code, unichar_id, permuter,
                 dawg_start: bool, word_start: bool, end: bool, dup: bool,
                 cert: float, score: float, prev, d, hash):
        self.code = code
        self.unichar_id = unichar_id
        self.score = score
        self.prev = prev
        self.code_hash = hash
        self.permuter = permuter
        self.start_of_dawg = dawg_start
        self.certainty = cert
        self.duplicate = dup
        self.start_of_word = word_start

class RecodeHeap:
    def __init__(self):
        self.elements: list[RecodeNode] = []

    def size(self):
        return len(self.elements)
    
    def add(self, node: RecodeNode):
        self.elements.append(node)
        self.elements.sort(key = lambda x: x.score, reverse = True)

    def set_and_reshuffle(self, i: int, node: RecodeNode):
        self.elements[i] = node
        self.elements.sort(key = lambda x: x.score, reverse = True)

    def peekTop(self):
        return self.elements[0]
    
    def pop(self):
        return self.elements.pop()

    def __repr__(self) -> str:
        return "HEAP(%s)"  %(str(list(map(lambda x: x.score, self.elements))))


class RecodeBeam:
    def __init__(self):
        self.beam_: dict[BeamIndex, RecodeHeap] = {} # Map<BeamIndex, RecodeHeap>

    def getBeam(self, index: BeamIndex):
        if index not in self.beam_:
            self.beam_[index] = RecodeHeap()
        return self.beam_[index]

class RecordBeamSearch:
    def __init__(self, recoder, null_char):
        self.kBeamWidths = [ 5, 10, 16, 16, 16, 16, 16, 16, 16, 16 ]
        # Default ratio between dict and non-dict words.
        self.kDictRatio = 2.25
        # Default certainty offset to give the dictionary a chance.
        self.kCertOffset = -0.085
        # Minimum value to output for certainty.
        self.kMinCertainty = -20.0
        # Probability corresponding to kMinCertainty.
        self.kMinProb = math.exp(self.kMinCertainty)

        self.is_simple_text_ = False

        self.recoder_ = recoder
        self.null_char_ = null_char
        self.beam_: list[RecodeBeam] = []
        #self.dict_ = None

    def decode(self, lstmData, worst_dict_cert, charset):
        for t in range(len(lstmData)):
            self.computeTopN(lstmData[t], self.kBeamWidths[0])
            self.decodeStep(lstmData[t], t, self.kDictRatio, self.kCertOffset, worst_dict_cert, charset)
        
    def computeTopN(self, outputs, topN):
        self.top_n_flags_ = [ TN_ALSO_RAN for i in range(len(outputs)) ]
        self.top_code_ = -1
        self.second_code_ = -1
        top_heap_ = []
        for i in range(len(outputs)):
            if len(top_heap_) < topN or outputs[i] > top_heap_[topN-1][0]:
                top_heap_.append((outputs[i], i))
                top_heap_.sort(key=lambda x: x[0], reverse=True)
                if len(top_heap_) > topN:
                    top_heap_.pop()
        while len(top_heap_) > 0:
            entry = top_heap_.pop()
            if len(top_heap_) > 1:
              self.top_n_flags_[entry[1]] = TN_TOPN
            else:
                self.top_n_flags_[entry[1]] = TN_TOP2
                if len(top_heap_) == 0:
                    self.top_code_ = entry[1]
                else:
                    self.second_code_ = entry[1]
        self.top_n_flags_[self.null_char_] = TN_TOP2

    def decodeStep(self, outputs, t: int, dict_ratio, cert_offset, worst_dict_cert, charset):
        if t == len(self.beam_):
            self.beam_.append(RecodeBeam())
        step = self.beam_[t]
        if t == 0:
            self.continueContext(None, BeamIndex(False, NC_ANYTHING, 0), outputs, TN_TOP2,
                            charset, dict_ratio, cert_offset, worst_dict_cert, step);
            # if self.dict_ != None:
            #     self.continueContext(None, BeamIndex(True, NC_ANYTHING, 0), outputs, TN_TOP2,
            #                 charset, dict_ratio, cert_offset, worst_dict_cert, step);
        else:
            prev = self.beam_[t - 1]
            total_beam = 0
            tn = 0
            while tn < TN_COUNT and total_beam == 0:
                top_n = tn
                for (index, heap) in prev.beam_.items():
                    for node in reversed(heap.elements):
                        self.continueContext(node, index, outputs, top_n,
                                        charset, dict_ratio, cert_offset, worst_dict_cert, step)
                for (index, heap) in step.beam_.items():
                    if index.cont == NC_ANYTHING:
                        total_beam += heap.size()
                tn += 1
            # for c in range(NC_COUNT):
            #     if step.best_initial_dawgs_[c].code >= 0:
            #         index = BeamIndex(True, c, 0)
            #         dawg_heap = step.beams_[index]
            #         self.pushHeapIfBetter(self.kBeamWidths[0], step.best_initial_dawgs_[c], dawg_heap)
        
    def continueContext(self, prev: RecodeNode, index: BeamIndex, outputs, top_n_flag, charset, dict_ratio, cert_offset, worst_dict_cert, step: RecodeBeam):
        previous = prev
        length = index.length
        prefix_codes = [ None for i in range(length) ]
        full_codes = [ None for i in range(length) ]
        use_dawgs = index.is_dawg
        prev_cont = index.cont
        for p in range(length - 1, -1, -1):
            previous = previous.prev
            while previous != None and (previous.duplicate or previous.code == self.null_char_):
                previous = previous.prev
            if previous != None:
                prefix_codes[p] = previous.code
                full_codes[p] = previous.code
        full_codes.append(0) # placeholder for current code
        full_code = RecodedCharID(1, full_codes)
        prefix = RecodedCharID(1, prefix_codes)
        if prev != None and not self.is_simple_text_:
            if self.top_n_flags_[prev.code] == top_n_flag:
                if prev_cont != NC_NO_DUP:
                    cert = self.probToCertainty(outputs[prev.code]) + cert_offset
                    self.pushDupOrNoDawgIfBetter(length, True, prev.code, prev.unichar_id,
                                cert, worst_dict_cert, dict_ratio, use_dawgs,
                                NC_ANYTHING, prev, step)
                if prev_cont == NC_ANYTHING and top_n_flag == TN_TOP2 and prev.code != self.null_char_:
                    cert = self.probToCertainty(outputs[prev.code] + outputs[self.null_char_]) + cert_offset
                    self.pushDupOrNoDawgIfBetter(length, True, prev.code, prev.unichar_id,
                                cert, worst_dict_cert, dict_ratio, use_dawgs,
                                NC_NO_DUP, prev, step)
            if prev_cont == NC_ONLY_DUP:
                return
            if prev.code != self.null_char_ and length > 0 and self.top_n_flags_[self.null_char_] == top_n_flag:
                cert = self.probToCertainty(outputs[self.null_char_]) + cert_offset;
                self.pushDupOrNoDawgIfBetter(length, False, self.null_char_, INVALID_UNICHAR_ID,
                              cert, worst_dict_cert, dict_ratio, use_dawgs,
                              NC_ANYTHING, prev, step)
        if prefix in self.recoder_.final_codes:
            final_codes = self.recoder_.final_codes[prefix]
            for code in final_codes:
                if self.top_n_flags_[code] != top_n_flag:
                    continue
                if prev is not None and prev.code == code and not self.is_simple_text_:
                    continue
                cert = self.probToCertainty(outputs[code]) + cert_offset
                if cert < self.kMinCertainty and code != self.null_char_:
                    continue
                full_code.codes[length] = code
                unichar_id = self.recoder_.decodeUnichar(full_code)
                if length == 0 and code == self.null_char_:
                    unichar_id = INVALID_UNICHAR_ID
                if unichar_id != INVALID_UNICHAR_ID and charset != None and False: # not charset.get_enabled(unichar_id):
                    continue
                self.continueUnichar(code, unichar_id, cert, worst_dict_cert, dict_ratio, use_dawgs, NC_ANYTHING, prev, step)
                if top_n_flag == TN_TOP2 and code != self.null_char_:
                    prob = outputs[code] + outputs[self.null_char_]
                    if prev and prev_cont == NC_ANYTHING and prev.code != self.null_char_ and (prev.code == self.top_code_ and code == self.second_code_ or code == self.top_code_ and prev.code == self.second_code_):
                        prob += outputs[prev.code]
                    cert = self.probToCertainty(prob) + cert_offset
                    self.continueUnichar(code, unichar_id, cert, worst_dict_cert, dict_ratio,
                                    use_dawgs, NC_ONLY_DUP, prev, step)

        if prefix in self.recoder_.next_codes:
            next_codes = self.recoder_.next_codes[prefix]
            raise Exception("TODO3")
    
    def continueUnichar(self, code, unichar_id, cert, worst_dict_cert, dict_ratio, use_dawgs, cont, prev: RecodeNode, step: RecodeBeam):
        if use_dawgs:
            if cert > worst_dict_cert:
                self.continueDawg(code, unichar_id, cert, cont, prev, step)
        else:
            nodawg_heap = step.getBeam(BeamIndex(False, cont, 0))
            self.pushHeapIfBetter(self.kBeamWidths[0], code, unichar_id, TOP_CHOICE_PERM, False,
                            False, False, False, cert * dict_ratio, prev, None, nodawg_heap)
            # if self.dict_ != None and ((unichar_id == UNICHAR_SPACE and cert > worst_dict_cert) or
            #         not self.dict_.getUnicharset().IsSpaceDelimited(unichar_id)):
            #     dawg_cert = cert # float
            #     permuter = TOP_CHOICE_PERM # PermuterType
            #     if unichar_id == UNICHAR_SPACE:
            #         permuter = NO_PERM;
            #     else:
            #         dawg_cert *= dict_ratio
            #     self.pushInitialDawgIfBetter(code, unichar_id, permuter, False, False,
            #                         dawg_cert, cont, prev, step);
    
    def probToCertainty(self, prob):
        if prob > self.kMinProb:
            return math.log(prob)
        return self.kMinCertainty
    
    def pushDupOrNoDawgIfBetter(self, length: int, dup: bool, code: int, unichar_id: int,
                               cert: float, worst_dict_cert: float, dict_ratio: float, use_dawgs: bool, cont, prev: RecodeNode, step: RecodeBeam):
        index = BeamIndex(use_dawgs, cont, length);
        if use_dawgs:
            if cert > worst_dict_cert:
              perm = NO_PERM
              if prev:
                perm = prev.permuter
              self.pushHeapIfBetter(self.kBeamWidths[length], code, unichar_id,
                       perm, False, False, False,
                       dup, cert, prev, None, step.getBeam(index))
        else:
            cert *= dict_ratio;
            if cert >= self.kMinCertainty or code == self.null_char_:
                perm = TOP_CHOICE_PERM
                if prev:
                    perm = prev.permuter 
                self.pushHeapIfBetter(self.kBeamWidths[length], code, unichar_id,
                       perm, False, False, False, dup, cert, prev, None, step.getBeam(index))

    
    def pushHeapIfBetter(self, max_size: int, code, unichar_id,
                        permuter, dawg_start: bool,
                        word_start: bool, end: bool, dup: bool,
                        cert: float, prev: RecodeNode,
                        d, heap: RecodeHeap):
        score = cert
        if prev:
            score += prev.score
        if heap.size() < max_size or score > heap.peekTop().score:
            hash = self.computeCodeHash(code, dup, prev)
            node = RecodeNode(code, unichar_id, permuter, dawg_start, word_start, end, dup, cert, score, prev, d, hash)
            if self.updateHeapIfMatched(node, heap):
                return
            heap.add(node)
            if heap.size() > max_size:
                heap.pop()

    def computeCodeHash(self, code, dup: bool, prev: RecodeNode):
        hash = 0
        if prev:
            hash = prev.code_hash
        if not dup and code != self.null_char_:
            num_classes = self.recoder_.code_range
            carry = ((hash >> 32) * num_classes) >> 32
            hash *= num_classes
            hash += carry
            hash += code
        return hash
    
    def updateHeapIfMatched(self, new_node: RecodeNode, heap: RecodeHeap):
        for (i, node) in enumerate(heap.elements):
            if node.code == new_node.code and node.code_hash == new_node.code_hash and node.permuter == new_node.permuter and node.start_of_dawg == new_node.start_of_dawg:
                if new_node.score > node.score:
                    heap.set_and_reshuffle(i, new_node)
                return True
        return False
    
    def extractBestPathAsWords(self, line_box, scale_factor, unicharset):
        (best_nodes, second_nodes) = self.extractBestPaths()
        #for node in best_nodes:
        #    print (node.code, node.code == self.null_char_, node.duplicate, node.unichar_id, unicharset[node.unichar_id])
        (unichar_ids, certs, ratings, xcoords, character_boundaries_) = self.extractPathAsUnicharIds(best_nodes)
        num_ids = len(unichar_ids)
        word_start = 0
        word_end = 0
        prev_space_cert = 0.0
        words = []
        while word_start < num_ids:
            word_end = word_start + 1
            while word_end < num_ids:
                if (unichar_ids[word_end] == UNICHAR_SPACE):
                    break
                index = xcoords[word_end];
                if best_nodes[index].start_of_word:
                    break
                #if best_nodes[index].permuter == TOP_CHOICE_PERM and (not unicharset.isSpaceDelimited(unichar_ids[word_end]) or not unicharset.isSpaceDelimited(unichar_ids[word_end - 1])):
                #    break
                word_end += 1

            space_cert = 0.0
            if word_end < num_ids and unichar_ids[word_end] == UNICHAR_SPACE:
                space_cert = certs[word_end]
            leading_space = word_start > 0 and unichar_ids[word_start - 1] == UNICHAR_SPACE
            word_res = ''
            for i in range(word_start, word_end):
                word_res += unicharset[unichar_ids[i]]
            words.append(word_res)
            prev_space_cert = space_cert
            if word_end < num_ids and unichar_ids[word_end] == UNICHAR_SPACE:
                word_end += 1
            word_start = word_end
        return words
    
    def extractBestPaths(self):
        best_node: RecodeNode = None
        second_best_node: RecodeNode = None
        last_beam = self.beam_[-1]
        for index in last_beam.beam_.keys():
            if index.cont == NC_ONLY_DUP:
                continue
            for node in last_beam.getBeam(index).elements:
                if index.is_dawg:
                    raise Exception("TODO")
                if best_node is None or node.score > best_node.score:
                    second_best_node = best_node
                    best_node = node
                if second_best_node is None or node.score > second_best_node.score:
                    second_best_node = node
        second_nodes = None
        if second_best_node:
            second_nodes = self.extractPath(second_best_node)
        best_nodes = self.extractPath(best_node)
        return (best_nodes, second_nodes)

    def extractPath(self, node: RecodeNode) -> "list[RecodeNode]":
        nodes = []
        while node:
            nodes.insert(0, node)
            node = node.prev
        return nodes
    
    def extractPathAsUnicharIds(self, best_nodes: "list[RecodeNode]"):
        unichar_ids = []
        certs = []
        ratings = []
        xcoords = []
        starts = []
        ends = []
        t = 0
        width = len(best_nodes)
        while t < width:
            certainty = 0.0
            rating = 0.0
            while t < width and best_nodes[t].unichar_id == INVALID_UNICHAR_ID:
                cert = best_nodes[t].certainty
                t += 1
                if cert < certainty:
                    certainty = cert
                rating -= cert
            starts.append(t)
            if t < width:
                unichar_id = best_nodes[t].unichar_id
                if unichar_id == UNICHAR_SPACE and len(certs) > 0 and best_nodes[t].permuter != NO_PERM:
                    if certainty < certs[-1]:
                        certs[-1] = certainty
                    ratings[-1] += rating
                    certainty = 0.0
                    rating = 0.0
                unichar_ids.append(unichar_id)
                xcoords.append(t)
                while True:
                    cert = best_nodes[t].certainty
                    t += 1
                    if cert < certainty or (unichar_id == UNICHAR_SPACE and best_nodes[t-1].permuter == NO_PERM):
                        certainty = cert
                    rating -= cert
                    if not (t < width and best_nodes[t].duplicate):
                        break
                ends.append(t)
                certs.append(certainty)
                ratings.append(rating)
            elif len(certs) > 0:
                if certainty < certs[-1]:
                    certs[-1] = certainty
                ratings[-1] += rating
        starts.append(width)
        character_boundaries = self.calculateCharBoundaries(starts, ends, width)
        xcoords.append(width)
        return (unichar_ids, certs, ratings, xcoords, character_boundaries)
    
    def calculateCharBoundaries(self, starts, ends, width):
        return None
