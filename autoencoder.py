from keras import Sequential as Seq
from keras.layers import Dense
from keras.layers import InputLayer
from keras.layers import Activation
from keras import losses
import numpy as np
import random
import sys
from PIL import Image

amount_to_train_on=10**0
number_of_epochs=200
amount_of_pictures_to_generate=20
folder_path_to_generated_output_files='GeneratedImages/'

def numberBetween255ClampedTo0And1(num):
	if(num>127):
		return 1
	return 0

def get_random_pixels_list(amount):
	return [random.uniform(0,255) for i in range(amount)]

# input_data_length = 64
# file_path_to_lossy_encode = sys.argv[1]
file_path_to_lossy_encode = 'TestAutoencoder2.png'

file_type_ending_to_lossy_encode=file_path_to_lossy_encode.split('.')[1]


img=Image.open(file_path_to_lossy_encode)

# currently only accepting PNG images
if(img.png ==None):
	print("Currently only accepting PNG images")
	exit(1)
# CONVERTING IMAGE TO MONOCHROME HERE
# img=img.convert('L')


# pixels=img.getdata()
# file_data=pixels
pixels=img.load()

# getting the pixels, similar to flatten or reshape a 2D array to 1D array
file_data=[]
for col in range(img.size[0]):
	for row in range(img.size[1]):
		# Appending the RGB values
		file_data.append(pixels[col,row][0])
		file_data.append(pixels[col,row][1])
		file_data.append(pixels[col,row][2])

# calculating the amount of all pixels of the inputted image it count as times 3 if we using RGB
input_data_length=len(file_data)

# bytes array to a list
# file_data=list(file_data)
# numpy the non-numpy array
file_np_array_data=np.array([file_data]*amount_to_train_on)


# file_np_array_data=file_np_array_data.flatten()



# building the autoencoder
autoencoder=Seq()
# 3 in input shape because of RGB
autoencoder.add(InputLayer(input_shape=(input_data_length,)))
# autoencoder.add(Dense((input_data_length*3)//4,activation='linear'))
autoencoder.add(Dense(input_data_length//32,activation='relu'))
autoencoder.add(Dense(input_data_length,activation='linear'))

autoencoder.compile(optimizer='adam',loss='mean_squared_error',metrics=['binary_accuracy'])

# sample data for training the autoencoder first
input_data=[
    get_random_pixels_list(input_data_length)
    for i in range(amount_to_train_on)]

# numpy the non-numpy array
input_data=np.array(input_data)


# first fitting the autoencoder
# autoencoder.fit(input_data,file_np_array_data,epochs=number_of_epochs)
autoencoder.fit(input_data,file_np_array_data,epochs=number_of_epochs)









# GENERATING the outputted files here


# generating random pixels noise
random_pixels_array=[
	get_random_pixels_list(input_data_length) for i in range(amount_of_pictures_to_generate)
]
# numpy the non-numpy array
random_pixels_array=np.array(random_pixels_array)


# actual generating here
predicted_outputs=autoencoder.predict(random_pixels_array)




# getting the current predicted output file that we want
predicted_output=predicted_outputs[0]
# getting the numpy array as a list
output_data=list(predicted_output)
# setting the new reconstructed image data
new_img=Image.new('RGB',img.size)

pixels=new_img.load()


for yIndex in range(new_img.size[0]):
	for xIndex in range(new_img.size[1]):
		# pixels[yIndex, xIndex] = int(output_data[(xIndex+yIndex*new_img.size[1])])
		# pixels[yIndex, xIndex] = numberBetween255ClampedTo0And1(output_data[(xIndex+yIndex*3*new_img.size[1])])
		new_pixel=(int(output_data[(xIndex+yIndex*new_img.size[1]+0)]),int(output_data[(xIndex+yIndex*new_img.size[1]+1)]),int(output_data[(xIndex+yIndex*new_img.size[1]+2)]))
		pixels[yIndex, xIndex] = new_pixel
		# pixels[yIndex, xIndex][0] = int(output_data[(xIndex+yIndex*new_img.size[1]+0)])
		# pixels[yIndex, xIndex][1] = int(output_data[(xIndex+yIndex*new_img.size[1]+1)])
		# pixels[yIndex, xIndex][2] = int(output_data[(xIndex+yIndex*new_img.size[1]+2)])


output_file_name='generated_file_'+str(0)+'.'+file_type_ending_to_lossy_encode
new_img.save(folder_path_to_generated_output_files+output_file_name,"PNG")



# for curFileOutputIndex in range(len(predicted_outputs)):
# 	# getting the current predicted output file that we want
# 	predicted_output=predicted_outputs[curFileOutputIndex]
# 	# getting the numpy array as a list
# 	output_data=list(predicted_output)
# 	# setting the new reconstructed image data
# 	new_img=Image.new('RGB',img.size)

# 	pixels=new_img.load()


# 	for yIndex in range(new_img.size[0]):
# 		for xIndex in range(new_img.size[1]):
# 			# pixels[yIndex, xIndex] = int(output_data[(xIndex+yIndex*new_img.size[1])])
# 			# pixels[yIndex, xIndex] = numberBetween255ClampedTo0And1(output_data[(xIndex+yIndex*3*new_img.size[1])])
# 			new_pixel=(int(output_data[(xIndex+yIndex*new_img.size[1]+0)]),int(output_data[(xIndex+yIndex*new_img.size[1]+1)]),int(output_data[(xIndex+yIndex*new_img.size[1]+2)]))
# 			pixels[yIndex, xIndex] = new_pixel
# 			# pixels[yIndex, xIndex][0] = int(output_data[(xIndex+yIndex*new_img.size[1]+0)])
# 			# pixels[yIndex, xIndex][1] = int(output_data[(xIndex+yIndex*new_img.size[1]+1)])
# 			# pixels[yIndex, xIndex][2] = int(output_data[(xIndex+yIndex*new_img.size[1]+2)])


# 	output_file_name='generated_file_'+str(curFileOutputIndex)+'.'+file_type_ending_to_lossy_encode
# 	new_img.save(folder_path_to_generated_output_files+output_file_name,"PNG")











