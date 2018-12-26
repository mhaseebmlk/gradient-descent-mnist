"""
Module for parsing the MNIST data as required in the problem statement
"""
import struct
import random
import gzip

import numpy as np

class Parser:
	def __init__(self,data_file_path,label_file_path):
		"""
		:param data_file_path: The local path of the data file. The file must be in .gz format.
		:param label_file_path: The local path of the corresponding label file. The file must be in .gz format.

		It is the user's responsibility to pass in the correct pair of data and label files.

		:rtype: None
		"""
		self.data_file_path=data_file_path
		self.labels_file_path=label_file_path

	def parse(self,feature_type,chunk_size=10000,shuffle=False,include_bias=False):
		"""
		:param chunk_size: Returns chunk_size-many examples from the input data. Default value is 10000, meaning it will return 10000 examples of the data depending on whether the user wants it shuffled or not.
		:param shuffle: If set to True, will shuffle the data before parsing. Default value is False.

		:rtype: array containing the parsed examples from the input data in the format [(example,lbl),...,(example,lbl)]
		"""

		unparsed_data = None
		parsed_data=[] 
		labels= None

		# print ('The feature type is:',feature_type)
		# print('Reading the data....')

		# Read the data 
		with gzip.open(self.data_file_path, 'rb') as f:
			zero, data_type, dims = struct.unpack('>HBB', f.read(4))
			shape = tuple(struct.unpack('>I', f.read(4))[0] \
	        	for d in range(dims))
			unparsed_data = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

		# Read the labels
		with gzip.open(self.labels_file_path, 'rb') as f:
			zero, data_type, dims = struct.unpack('>HBB', f.read(4))
			shape = tuple(struct.unpack('>I', f.read(4))[0] \
	        	for d in range(dims))
			labels = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

		combined_data=zip(unparsed_data,labels)

		if shuffle==True:
			random.Random(len(combined_data)).shuffle(combined_data)

		num_features_org=28 # the original image is 28x28
		parsed_data=list()

		if feature_type=='type1':

			num_features_parsed=785 if include_bias==True else 784

			for i in range(chunk_size):
				img=np.zeros(num_features_parsed)
				idx=0

				if include_bias==True:
					img[0]=np.random.rand()  # set the bias
					idx=1

				for j in range(num_features_org):
					for k in range(num_features_org):
						old_val=combined_data[i][0][j][k]
						new_val=old_val/255.0
						img[idx]=new_val
						idx+=1

				parsed_data.append((img,combined_data[i][1]))

		elif feature_type=='type2':
			num_features_parsed=197 if include_bias==True else 196
			
			for i in range(chunk_size):
				org_img=combined_data[i][0]
				img=np.ones(num_features_parsed)
				idx=0

				if include_bias==True:
					img[0]=np.random.rand()  # set the bias
					idx=1

				for j in range(0,num_features_org,2):
					for k in range(0,num_features_org,2):
						cur=org_img[j][k]
						right=org_img[j][k+1]
						bottom=org_img[j+1][k]
						bottom_right=org_img[j+1][k+1]
						# print (idx,j,k,cur,right,bottom,bottom_right)
						old_val=max(cur,right,bottom,bottom_right)
						new_val=old_val/255.0
						img[idx]=new_val
						idx+=1

				parsed_data.append((img,combined_data[i][1]))
		else:
			raise Exception('Invalid feature type:',feature_type)

		return parsed_data