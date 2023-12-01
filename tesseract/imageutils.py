
from PIL import Image

def get_component(data, compIndex):
    return [ [ v[compIndex] for v in line ] for line in data ]

def save_to_image(name, data):
    def clip(v):
        return max(min(int(v), 255), 0)
    im = Image.new(mode="L", size=(len(data[0]), len(data)))
    minv = None
    maxv = None
    for y in range(im.height):
        for x in range(im.width):
            v = data[y][x]
            if minv is None:
                minv = v
                maxv = v
            else:
                minv = min(minv, v)
                maxv = max(maxv, v)
    contrast = (maxv - minv) / 2.0
    def map(x):
        if contrast == 0:
            return 0
        return (x - minv) / contrast - 1.0
    for y in range(im.height):
        for x in range(im.width):
            im.putpixel((x, y), clip((map(data[y][x]) + 1.0) * 128))
    im.save(name + ".png")

def save_to_image_line_contrast(name, data):
    def clip(v):
        return max(min(int(v), 255), 0)
    im = Image.new(mode="L", size=(len(data[0]), len(data)))
    for y in range(im.height):
        minv = None
        maxv = None
        for x in range(im.width):
            v = data[y][x]
            if minv is None:
                minv = v
                maxv = v
            else:
                minv = min(minv, v)
                maxv = max(maxv, v)
        contrast = (maxv - minv) / 2.0
        def map(x):
            if contrast == 0:
                return 0
            return (x - minv) / contrast - 1.0
        for x in range(im.width):
            im.putpixel((x, y), clip((map(data[y][x]) + 1.0) * 128))
    im.save(name + ".png")

def transpose(data):
    return [ [ data[i][j] for i in range(len(data)) ] for j in range(len(data[0])) ]