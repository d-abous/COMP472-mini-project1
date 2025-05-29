import struct
import numpy as np

"""
number of pixels X number of immages = bytes of pixel data

frombuffer stores pixle data into a 1d numpy array
 * all in one line
 * just converts raw bytes to numpy array

.reshape(num, rows*cols) turns that 1d array into a 2d array of individual images 
 * num = number of images 
 * rows*cols = number of pixels per image. 28 pixels high x 28 pixels wide → (28, 28) shape. each pizel has a value (0,255)
 * Turn each 28x28 image into a 1D array with 784 pixel values

 this is what the 2d array looks like: 
 Pixel 0	Pixel 1	 ...	Pixel 783 
    0	        255	 ...	   23         ... each row is an image. each column is a pixel value 

Each individual image is a 1D array (784 pixels)- flattening (turning each 28x28 image into 784 pixels in a row)

The entire dataset is a 2D array (shape: 60000 x 784) 


"""



def load_images(filepath): #when you call this method you pass a filepath as a string like: load_images("train-images.idx3-ubyte")
    with open (filepath, 'rb') as f: # rb for binary files. f is a file object
     
     #reads the first 16 bites that is the file header - gives us info about the file so the program knows what data to expect 
     # I 's are for number of integers that correspond to bytes. this is MNIST format
     # there are 4 initgers (I 's) that tell us magic (file type), num, rows, cols
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # prints the data we just read from
        print(f"Loaded {num} images with size {rows}x{cols}")
        # read the rest of the binary file 
        # np.frombuffer converts it into a NumPy array
        # they array is of type uint8 (0 to 255 black-white)
        data = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshapes the flat array into a 2D array REVIEW THIS 
        data = data.reshape(num, rows * cols)

        #Returns the processed image data as a NumPy array of shape (num, 784)
    return data

# if __name__=="__main__":
#    load_images()

def load_labels(filepath):

    #open file in binary read mode as an object f
    with open(filepath, 'rb') as f:
        # read the file header 
        magic, num = struct.unpack(">II", f.read(8))
        # print of how many labels were loaded
        print(f"Loaded {num} labels")
        #Reads the remaining bytes of the file.
        #Each label is a single unsigned 8-bit integer (0–9), corresponding to an image.
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        # Returns the labels as a 1D NumPy array of length num.
    return labels

