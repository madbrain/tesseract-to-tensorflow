
from tesseract.datareader import read_components, Recognizer
from tesseract.preprocess import preprocess_image
from tesseract.network import apply_network
from tesseract.beamsearch import RecordBeamSearch, kWorstDictCertainty, kCertaintyScale

components = read_components("/usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata")
recognizer = Recognizer()
recognizer.read(components)

filename = "data/lines.jpg"
lineBox = (44, 25, 1109, 81)
#lineBox = (44, 96, 1109, 154)
#lineBox = (44, 166, 839, 222)

inputData = preprocess_image(filename, lineBox)
r = apply_network(recognizer.network, inputData)

beamSearch = RecordBeamSearch(recognizer.recoder, recognizer.null_char)
beamSearch.decode(r, kWorstDictCertainty / kCertaintyScale, recognizer.charset)

line_box = None
scale_factor = 1.0

words = beamSearch.extractBestPathAsWords(line_box, scale_factor, recognizer.charset)
print (words)