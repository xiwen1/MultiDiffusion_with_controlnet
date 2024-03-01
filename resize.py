from PIL import Image 
import numpy as np 


image = Image.open('simp_house_sketch.jpg')
image = image.resize((768, 512))
image.save('simp_house_sketch_resized.jpg')