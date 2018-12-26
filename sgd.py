import sys
import random
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from parser import Parser

class StochasticGradientDescent:
	def __init__(self,training_set,testing_set,num_epochs,learning_rate,feature_type,lmbda=None):
		assert len(training_set) != 0, 'Error: training set cannot be empty.'
		assert len(testing_set) != 0, 'Error: testing set cannot be empty.'

		self.training_data=training_set
		self.testing_data=testing_set

		random.shuffle(self.training_data)

		self.num_epochs=num_epochs
		self.feature_type=feature_type
		self.learning_rate=learning_rate

		self.regularization = False
		self.lmbda = None
		if lmbda!=None:
			self.regularization = True
			self.lmbda = lmbda

		self.num_features=len(self.training_data[0][0])
		self.num_digits=10

		self.weights=[np.zeros(self.num_features) for i in range(self.num_digits)]

	def train(self,graph=True):
		avg_training_losses=list()
		training_accuracies=list()
		test_accuracies=list()
		epochs=list()
		e=0
		interval_sz = 100

		while e < self.num_epochs:
			for img, lbl in self.training_data:
				for i in range(len(self.weights)):
					delta_w = np.zeros(self.num_features)
					digit=i
					y=1 if digit==lbl else 0 
					act=np.dot(self.weights[digit],img)
					for j in range(self.num_features):
						delta_w[j] = self.learning_rate * (y - self.logistic(act)) * img[j]

					if self.regularization == True:
						delta_w += self.lmbda * self.weights[digit]
					self.weights[digit] += delta_w

				if e%interval_sz == 0:
					
					avg_training_loss = self.get_training_loss(self.training_data,self.weights)
					avg_training_losses.append(avg_training_loss)

					training_accuracy = self.get_accuracy('training')
					test_accuracy=self.get_accuracy('test')

					training_accuracies.append(training_accuracy)
					test_accuracies.append(test_accuracy)

					epochs.append(e)

					print ('epoch: {}, Training loss: {}, Training Accuracy: {}, Test Accuracy: {}'.format(e,avg_training_loss,training_accuracy,test_accuracy))

				e+=1
				if e == self.num_epochs: # stopping criterion
					break

		# # num epochs tuning/ info for graphing
		# print ('avg_training_losses: ',avg_training_losses)
		# print ('training_accuracies: ',training_accuracies)
		# print ('epochs: ',epochs)
		# print ('test_accuracies: ',test_accuracies)

		if graph==True:
			# name = 'convergence_{}_{}_{}.png'.format('SGD',self.regularization,self.feature_type)
			name = 'convergence.png'
			self.graph(name,epochs,training_accuracies,test_accuracies)

		return epochs,test_accuracies

	def graph(self,name,epochs,training_accuracies,test_accuracies):
		fig, ax = plt.subplots()
		line1, = ax.plot(epochs, training_accuracies, label='train')
		line1.set_dashes([2, 2, 10, 2]) 
		line2, = ax.plot(epochs, test_accuracies, label='test')
		ax.legend()
		title='Algorithm: {}, Regularization: {}, Feature Type: {}'.format('SGD',self.regularization,self.feature_type)
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.title(title)
		plt.savefig(name)

	def get_training_loss(self,data, perceptrons):
		avg_training_loss=0.0
		training_losses=list()
		for digit in range(len(self.weights)):
			training_loss=0.0
			for img, lbl in self.training_data:
				act=np.dot(self.weights[digit],img)
				logistic_ = self.logistic_2(act)
				if logistic_==0:
					logistic_ = 0.00001
				elif logistic_==1:
					logistic_ = 0.99999
				y=1 if digit==lbl else 0 
				training_loss += (y*math.log(logistic_)) + ((1-y)*math.log(1-logistic_))  
			training_losses.append(-training_loss)
		avg_training_loss = (sum(training_losses))/(self.num_digits*len(self.
			training_data)+0.0)
		return avg_training_loss

	def get_accuracy(self,type_):
		data=None
		if type_=='training': 
			data = self.training_data 
		elif type_ == 'test':
			data = self.testing_data
		else:
			raise Exception('Invalid accuracy type:',type_)

		accuracy=0.0
		for img, lbl in data:
			predicted_digit=self.predict(img)
			if predicted_digit==lbl:
				accuracy+=1
		accuracy=(accuracy/len(data)) * 100
		return accuracy

	def logistic(self,z): return 0.5 + 0.5*(np.tanh(z/2.0))

	def logistic_2(self,z): return 1.0 / (1.0 + np.exp(-z))

	def predict(self,img):
		acts=[np.dot(self.weights[digit],img) for digit in range(len(self.weights))]
		predicted_digit=acts.index(max(acts))
		return predicted_digit

def tune_lambda(training_data,test_data):
	'''
	{
		sgd_True_type1_1500000:9000.0
		sgd_True_type2_1500000:5900.0
	}
	'''
	lambdas = [0.0001,0.001,0.005,0.01,0.05,0.1,0.2]
	# lambdas = [0.0001,0.001]
	colors=['blue','green','red','cyan','magenta','yellow','black']
	feature_types = ('type1','type2')

	learning_rate = 0.1	
	for ft in feature_types:
		num_epochs = None
		if ft=='type1':
			num_epochs=9100
		elif ft=='type2':
			num_epochs=6300

		name='tuning_{}_{}.png'.format('SGD',ft)
		fig, ax = plt.subplots()
		title='Tuning Lambda. Algorithm: {}, Feature Type: {}'.format('SGD',ft)
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')
		plt.title(title)
		
		for i,l in enumerate(lambdas):
			stochastic_gd = StochasticGradientDescent(training_data,test_data,num_epochs,learning_rate,ft,l) 
			eps,test_accuracies = stochastic_gd.train(graph=False)
			line2, = ax.plot(eps, test_accuracies,color=colors[i],label='lambda = {}'.format(l))
			ax.legend()
		plt.savefig(name)

def print_usage():
	print('Usage:')
	print('python <filename>.py [regularization? True/False] [Feature_Type? type1/type2] [path to DATA_FOLDER]')

def check_cmd_line_args():
	if len(sys.argv) != 4:
		print_usage()
		sys.exit()

def read_cmd_line_args():
	regularization,feature_type,data_folder = (sys.argv[1]),sys.argv[2],sys.argv[3]
	if (regularization not in ('True','False')) or (feature_type not in ('type1','type2')):
		print_usage()
		sys.exit()
	return regularization,feature_type,data_folder

def main():
	check_cmd_line_args()
	regularization,feature_type,data_folder=read_cmd_line_args()

	train_data=data_folder+'/'+'train-images-idx3-ubyte.gz'
	train_label_data=data_folder+'/'+'train-labels-idx1-ubyte.gz'
	training_parser=Parser(train_data,train_label_data)
	training_parsed=training_parser.parse(feature_type=feature_type,chunk_size=10000,include_bias=False)

	test_data=data_folder+'/'+'t10k-images-idx3-ubyte.gz'
	test_label_data=data_folder+'/'+'t10k-labels-idx1-ubyte.gz'
	testing_parser=Parser(test_data,test_label_data)
	testing_parsed=testing_parser.parse(feature_type=feature_type,include_bias=False)

	'''
	{
		lambda: 0.001
		sgd_False_type1_1500000:20100,
		sgd_False_type2_1500000:59900.0
		sgd_True_type1_1500000:9000.0
		sgd_True_type2_1500000:5900.0
		}
	'''
	num_epochs = None
	lmbda = 0.001
	if regularization=='False' and feature_type=='type1':
		num_epochs=20200
	elif regularization=='False' and feature_type=='type2':
		num_epochs=60000
	elif regularization=='True' and feature_type=='type1':
		num_epochs=9100
	elif regularization=='True' and feature_type=='type2':
		num_epochs=6300

	learning_rate = 0.1	

	stochastic_gd = None
	if regularization=='False': 
		stochastic_gd = StochasticGradientDescent(training_parsed,testing_parsed,num_epochs,learning_rate,feature_type) 
	else:
		stochastic_gd = StochasticGradientDescent(training_parsed,testing_parsed,num_epochs,learning_rate,feature_type,lmbda) 
	stochastic_gd.train()

	# tune_lambda(training_parsed,testing_parsed)

if __name__ == '__main__':
	main()