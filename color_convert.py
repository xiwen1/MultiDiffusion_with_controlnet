from PIL import Image 
import numpy as np

image = Image.open('simp_house_sketch_resized.jpg')
arr = np.array(image)
arr = 255 - arr
image = Image.fromarray(arr)
image.save('simp_house_sketch_converted.jpg')