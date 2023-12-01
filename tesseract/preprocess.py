
from PIL import Image, ImageOps

from tesseract.constants import TARGET_HEIGHT

class Stats:
    def __init__(self, min, max):
        self.min = min
        self.max = max
        self.elements = [ 0 for i in range(max - min) ]
        self.totalCount = 0
    
    def add(self, value, count):
        self.elements[value - self.min] += count
        self.totalCount += count

    def ile(self, frac):
        target = self.totalCount * frac
        sum = 0
        index = None
        for (i, v) in enumerate(self.elements):
            if sum >= target:
                index = i
                break
            sum += v
        if index is not None:
            return self.min + index - (sum - target) / float(self.elements[index-1])
        return float(self.min)

def compute_black_white(image):
    mins = Stats(0, 256)
    maxes = Stats(0, 256)
    if len(image[0]) > 3:
        # sample middle line for local minimums and maximums
        y = len(image) // 2
        prev = image[y][0]
        curr = image[y][1]
        for x in range(2, len(image[0])):
            next = image[y][x]
            if curr < prev and curr <= next or curr <= prev and curr < next:
                mins.add(curr, 1)
            if curr > prev and curr >= next or curr >= prev and curr > next:
                maxes.add(curr, 1)
            prev = curr
            curr = next

    if mins.totalCount == 0:
        mins.add(0, 1)
    if maxes.totalCount == 0:
        maxes.add(255, 1)
    black = mins.ile(0.25)
    white = maxes.ile(0.75)
    return (black, white)

def convert_to_input(image):
    imageData = [ [ image.getpixel((x, y)) for x in range(image.width) ] for y in range(image.height) ]
    (black, white) = compute_black_white(imageData)
    contrast = (white - black) / 2
    if contrast <= 0:
        contrast = 1
    def to_pixel(x, y):
        return (imageData[y][x] - black) / contrast - 1.0
    return [ [ to_pixel(x, y) for x in range(len(imageData[0])) ] for y in range(len(imageData)) ]


def preprocess_image(filename, lineBox):
    inputImage = Image.open(filename)

    print("> Preprocess image")
    grayImage = ImageOps.grayscale(inputImage)
    lineImage = grayImage.crop(lineBox)
    inputImage = ImageOps.scale(lineImage, float(TARGET_HEIGHT) / lineImage.height)
    inputImage.save("line.png")

    return convert_to_input(inputImage)