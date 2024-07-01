#对于单个cable衰减，lr取0.0001和0.0005时效果更好
#filters=256, kernel_size=5, strides=1,
#已经打乱了数据集和标签的次序，为什么还是会把所有数据集归为一类呢？？？
#答案：当数据集的内容进行了normalization之后，准确率大幅提升，发现不同程度刚度衰减（一个衰减20%一个衰减60%）二分类已经可以达到100%准确率了
#filters=256, kernel_size=5，初始学习率0.00001发现可以识别二分类的同程度刚度衰减（都衰减60%），准确率可达到100%
#filters=256, kernel_size=5，dropout0.5,初始学习率0.00001发现也可以识别五分类的同程度刚度衰减，准确率在50epoch后可以达到100%
#trainX, testX, trainy, testy = train_test_split(trainX,trainy,test_size=0.2,random_state=1)
from numpy import mean
from numpy import std
from numpy import dstack
from numpy import hstack
from numpy import vstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import os
import pandas as pd
import numpy as np
from keras.layers import BatchNormalization
from keras.callbacks import ReduceLROnPlateau
import keras
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras import optimizers
from keras import callbacks
from keras.callbacks import TensorBoard
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import train_test_split



# 经过运算发现原网页的代码运行后val的acc也到了90%以上，而本代码的val-acc只有70%，还是要进一步优化
# load a single file as a numpy array把该路径文件的内容return成一个数组
def load_file1(filepath):#用于读取label
	# 仅用read_csv会导致只读取excel中的一列，只用open（）会导致第一行被默认为标题，因此用嵌套，把open（）作为read_csv的filepath
	# 然后再借助read_csv的header=None命令第一行不是标题，这样就可以两全其美了
	dataframe = pd.read_csv(open(filepath, 'r'), header=None)
	return dataframe#.values

def load_file(filepath):#只对于读取训练集文件
	# 仅用read_csv会导致只读取excel中的一列，只用open（）会导致第一行被默认为标题，因此用嵌套，把open（）作为read_csv的filepath
	# 然后再借助read_csv的header=None命令第一行不是标题，这样就可以两全其美了
	dataframe = pd.read_csv(open(filepath, 'r'), header=None)
	data_array = dataframe.to_numpy()
	euclidean_norms = np.sqrt(np.sum(data_array**2, axis=0))
	normalized_data_array = data_array / euclidean_norms
	normalized_data = pd.DataFrame(normalized_data_array)
	#print(normalized_data)
	return normalized_data#.values


# load a list of files and return as a 3d numpy array，把多个文件中return的二维数组整合到一个三维数组中
# 这里的prefix会在下文定义文件所在路径
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		loaded.append(data)
		#print(loaded)
	# stack group so that features are the 3rd dimension

	loaded = dstack(loaded)
	loaded = loaded.swapaxes(2,0)
	loaded = loaded.swapaxes(2,1)
	#loaded = loaded.squeeze()压缩所有为1的维度
	print(loaded)



	return loaded


# load a dataset group, such as train or test，这里输出的值X和y就分别是input信号（采集的加速度信号）和output信号（每个信号的label）
# 这里相当于对于目标文件夹先有几个子文件夹state0，state1.。。通过这几个子文件夹创建一个list按照顺序储存文件名，然后在子文件夹外再放置这些excel来提供文件所在路径
# 否则总会出现各种错误，如排序不对，或者路径不对等等
def load_dataset_group(group, label_group, prefix=''):
	filepath = prefix + group + '/'
	# load all 9 files as a single array
	# filenames = list()
	# total acceleration
	# 这里os.listdir是把指定路径下的文件名都存入一个列表，但排序方式是先把1,10,100,1000拿出来排序，然后再排其它的，这是不对的，所以要对列表进行重排序
	# 不知道为什么合并之后第二个维度是1，按理说第二个维度应该是15啊？
	#filenames0 = os.listdir(prefix + group + '/' + 'state0-10/')
	#filenames0.sort(key=lambda x: int(x[7:-4]))
	filenames1 = os.listdir(prefix + group + '/' + 'state0/')
	filenames1.sort(key=lambda x: int(x[3:-4]))
	filenames2 = os.listdir(prefix + group + '/' + 'state1/')
	filenames2.sort(key=lambda x: int(x[3:-4]))
	filenames3 = os.listdir(prefix + group + '/' + 'state2/')
	filenames3.sort(key=lambda x: int(x[3:-4]))
	filenames4 = os.listdir(prefix + group + '/' + 'state3/')
	filenames4.sort(key=lambda x: int(x[3:-4]))
	filenames5 = os.listdir(prefix + group + '/' + 'state4/')
	filenames5.sort(key=lambda x: int(x[3:-4]))
	filenames6 = os.listdir(prefix + group + '/' + 'state5/')
	filenames6.sort(key=lambda x: int(x[3:-4]))
	filenames7 = os.listdir(prefix + group + '/' + 'state6/')
	filenames7.sort(key=lambda x: int(x[3:-4]))
	filenames8 = os.listdir(prefix + group + '/' + 'state7/')
	filenames8.sort(key=lambda x: int(x[3:-4]))
	filenames9 = os.listdir(prefix + group + '/' + 'state8/')
	filenames9.sort(key=lambda x: int(x[3:-4]))
	filenames10 = os.listdir(prefix + group + '/' + 'state9/')
	filenames10.sort(key=lambda x: int(x[3:-4]))
	filenames11 = os.listdir(prefix + group + '/' + 'state10/')
	filenames11.sort(key=lambda x: int(x[4:-4]))
	# filenames2 = os.listdir(prefix + group + '/' + 'state2-50/')
	# filenames2.sort(key=lambda x: int(x[7:-4]))
	# filenames3 = os.listdir(prefix + group + '/' + 'state3-70/')
	# filenames3.sort(key=lambda x: int(x[7:-4]))
	# filenames4 = os.listdir(prefix + group + '/' + 'state4-90/')
	# filenames4.sort(key=lambda x: int(x[7:-4]))
	#filenames5 = os.listdir(prefix + group + '/' + 'state5-health/')
	#filenames5.sort(key=lambda x:int(x[7:-4]))
	filenames = filenames1 + filenames2 + filenames3 + filenames4 + filenames5 + filenames6 + filenames7 + filenames8 + filenames9 + filenames10 + filenames11
	print(filenames)
	# filenames.sort()
	# load input data
	X = load_group(filenames, filepath)
	# load class output这里是调用y_test文件，这个文件就是一个一列的txt文件，表明了每个信号的label
	y = load_file1(prefix + label_group + '/y_' + group + '.txt')
	return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
	# load all train，相当于下面trainX就是上面return的X，trainy就是return的y
	# trainX对应了输入的input数据，trainy对应了输出的分类label，如本代码print的trainy.shape就是（7352,1），就是给每个采集的数据添加label
	trainX, trainy = load_dataset_group('state_train', 'label_train', prefix)
	#print(trainX)
	print(trainX.shape, trainy.shape)
	# load all test
	#testX, testy = load_dataset_group('state_test', 'label_test', prefix)
	#随机分配20%给test
	trainX, testX, trainy, testy = train_test_split(trainX,trainy,test_size=0.2,random_state=0)
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1
	# one hot encode y，进行热编码，是把trainy和testy在二进制和十进制之间转换
	# 原本trainy和testy就是含有label的txt文件，只有一列（包含六种不同的分类情况），热编码后则转变成有六列的矩阵
	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)
	# 同时打乱训练集和label的顺序，以提高CNN的泛化能力
	# permutation = np.random.permutation(trainX.shape[0])
	# permutation_test = np.random.permutation(testX.shape[0])
	# shuffled_dataset = trainX[permutation, :, :]
	# shuffled_labels = trainy[permutation]
	# shuffled_dataset_test = testX[permutation_test, :, :]
	# shuffled_labels_test = testy[permutation_test]
	# trainX = shuffled_dataset
	# trainy = shuffled_labels
	# testX = shuffled_dataset_test
	# testy = shuffled_labels_test
	#print(trainX)
	#print(trainy)
	print(testy)
	print(trainy)
	return trainX, trainy, testX, testy
#下面这行如果加上了会导致多显示一次load_dataset里的那四行
#trainX, trainy, testX, testy = load_dataset('D:/dataset/Python1D_CNNs-master/')

#def scheduler(epoch):
#	lr = [0.01, 0.01, 0.001, 0.0001]
#	epochs = [5, 10, 15, 19, 23]
#	if epoch in epochs:
#		index_lr = epochs.index(epoch)
#		lr_now = lr_inputs[index_lr]
#		lr = K.get_value(model.optimizer.lr)
#		K.set_value(model.optimizer.lr, lr_now)
#		print("pre_lr {}".format(lr))
#		print("lr changed to {}".format(lr_now))
#	return K.get_value(model.optimizer.lr)
# fit and evaluate a model
#def get_lr_metric(optimizer):
#	def lr(y_true, y_pred):
#		return optimizer.lr
#	return lr
def identity_block(X,filters):
	## 第一种残差学习
	# 定义基本名字
	# con_name_base = 'res' + str(stage) + block + '_branch'
	# bn_name_base = 'bn' + str(stage) + block + '_branch'
	F1, F2, F3 = filters
	# 保存输入值
	X_shortcut = X
	# 残差主路径第一部分
	X = Conv1D(filters=F1, kernel_size=1, padding='same')(X)
	# BatchNormalization能让学习的泛化能力更强，但是axis取多少要注意
	X = BatchNormalization(axis=1)(X)
	X = Activation('relu')(X)
	# 残差主路径第二部分
	X = Conv1D(filters=F2, kernel_size=3, padding='same')(X)
	X = BatchNormalization(axis=1)(X)
	X = Activation('relu')(X)
	# 残差主路径第三部分
	X = Conv1D(filters=F3, kernel_size=1, padding='same')(X)
	X = BatchNormalization(axis=1)(X)
	# 给主路径添加shortcut部分，并用relu激活
	X = layers.add([X, X_shortcut])
	X = Activation('relu')(X)
	return X
def convolutional_block(X,filters):
	## 第二种残差学习
	# 定义基本名字
	# con_name_base = 'res' + str(stage) + block + '_branch'
	# bn_name_base = 'bn' + str(stage) + block + '_branch'
	F1, F2, F3 = filters
	# 保存输入值
	X_shortcut = X
	# 残差主路径第一部分
	X = Conv1D(filters=F1, kernel_size=1, padding='same')(X)
	X = BatchNormalization(axis=1)(X)
	X = Activation('relu')(X)
	# 残差主路径第二部分
	X = Conv1D(filters=F2, kernel_size=3, strides=2, padding='same')(X)
	X = BatchNormalization(axis=1)(X)
	X = Activation('relu')(X)
	# 残差主路径第三部分
	X = Conv1D(filters=F3, kernel_size=1, padding='same')(X)
	X = BatchNormalization(axis=1)(X)
	# shortcut路径，该部分的filters总是与主路径最后一部分的filters相等
	X_shortcut = Conv1D(filters=F3, kernel_size=1, strides=2, padding='same')(X_shortcut)
	X_shortcut = BatchNormalization(axis=1)(X_shortcut)
	# 给主路径添加shortcut部分，并用relu激活
	X = layers.add([X, X_shortcut])
	X = Activation('relu')(X)
	return X
def evaluate_model(trainX, trainy, testX, testy, n_filters):#把参数作为输入参数来控制每次调用该函数时使用的超参数不同
	# verbose取0，在控制台没有任何输出，若取1，则会显示训练的进度条
	# epochs取10就是对所有数据集训练了十次（分为十代训练）
	# batch_size取32代表在训练时每次取一组样本（一个batch）所包含的样本数，往往取2的幂。通常网络较小时batch_size大一些
	# train这个CNN时，共7352个样本，batch_size是32，则可分为230个batch（其中有229个batch包含32个样本，最后一个batch包含24个样本），总共对所有样本训练10代 （epochs为10）
	verbose, epochs, batch_size = 1, 70, 128
	# trainX.shape[1]就代表取trainX的第二个元素，trainX.shape是（7352,128,9），则取128
	# 同理，trainX.shape[2]取第三个元素，即9。而trainy.shape[1]是（7352,6）的第二个元素，就是6
	# 因此trainX.shape[1]代表时间步长128，trainX.shape[2]代表特征个数（每个信号包含9个输入，分为body accel，total accel和body gyroscope，每个都包含三个轴的数据），trainy.shape[1]代表输出标签的种类6种
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	model = Sequential()
	#filter就是卷积核，也就是所谓的特征feature，filters是64表示选择了64个特征，对每个特征分别进行一次卷积，得到64个feature map
	#卷积核filter的尺寸就是kernel_size，也就是特征像素块的大小
	# filters=64代表着用了64个并行的卷积窗口，kernel_size=3代表了卷积核大小（每个窗口的尺寸）为3
	# 就是说进行64次卷积，每次卷积的卷积核大小都是3，最终能得到64个feature map。filters（取2的幂）和kernel_size（取2,3,5,7,11）可以更改，来看看更改后如何提高模型准确率
	# 卷积层作为第一层的时候要给出input_shape()，给出时间步长和信号个数，也就是给出输入数据是几行几列的
	model.add(Conv1D(filters=256, kernel_size=7, strides=1, padding='same', activation='relu', input_shape=(n_timesteps, n_features)))
	model.add(Conv1D(filters=256, kernel_size=7, padding='same', activation='relu'))
	#model.add(Conv1D(filters=n_filters, kernel_size=3, padding='same', activation='relu'))
	# Residual_1 = identity_block(layer1, filters=[64, 64, 128])
	# Residual_2 = convolutional_block(Residual_1, filters=[128, 128, 256])
	# Residual_3 = identity_block(Residual_2, filters=[128, 128, 256])
	# Residual_4 = convolutional_block(Residual_3, filters=[256, 256, 512])
	# Residual_5 = identity_block(Residual_4, filters=[256, 256, 512])
	# 输出是二维的，输入是三维的，因此这样需要对结果进行一下降维再输入到dense层
	# output_res = Flatten()(Residual_5)
	# model.add(output_res)
	# model.add(BatchNormalization(axis=2))
	# model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='sigmoid'))
	# 有的说val准确率不高是下面这个BN的问题，要把其中源代码的training设置为training=False，删掉试试看。发现删掉后val的准确率更低了，下面试试看加一个training=False，改完准确率越来越低
	# axis取1时最大val的acc是0.63，axis取2时有0.69，取3就报错了
	model.add(BatchNormalization(axis=2))
	model.add(MaxPooling1D(pool_size=3))
	model.add(Flatten())
	# dropout（0.5）就是每次迭代让一半卷积核不更新，这样可以防止过拟合现象
	model.add(Dropout(n_filters))

	# flatten是一个降维过程（默认按行降维），如a=([[1,2],[3,4],[5,6]])，a.flatten()就=([1,2,3,4,5,6])
	# flatten('F')是按列降维，flatten('A')是按行降维，则a.flatten('F')=([1,3,5,2,4,6])

	# 下式是全连接层的输入层，dense就是全连接层，100代表着unit，也就是全连接层有100个元素组成
	model.add(Dense(256, activation='relu'))
	# 下式是全连接层的输出层，n_outputs值是6（分为六类标签），就是说该输出层由六个元素组成，再经过softmax转化成每种分类的概率
	model.add(Dense(n_outputs, activation='softmax'))
	plot_model(model, show_shapes=True, to_file=f'multichannel{n_filters}.png')
	# 用adam
	adam = keras.optimizers.Adam(lr=0.00018, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0)
	#lr_metric = get_lr_metric(adam)
	#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', lr_metric])
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	model.get_weights()#查看权重
	# 用Learning rate scheduler
	#def scheduler(eposh):
	#	if epoch % 5 ==0 and epoch != 0:
	#		lr = K.get_value(model.optimizer.lr)
	#		K.set_value(model.optimizer.lr, lr * 0.5)
	#		print("lr changed to {}".format(lr * 0.5))
	#	return K.get_value(model.optimizer.lr)
	#reduce_lr = learningRateScheduler(scheduler)

	# fit network
	# model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
	# evaluate model
	# print(eval(model.optimizer.lr))
	# reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
	# history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size,
	#					verbose=verbose, validation_data=(testX, testy), callbacks=[reduce_lr])
	history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size,
						verbose=verbose, validation_data=(testX, testy))
	# 绘制混淆矩阵
	plot_confuse(model, testX, testy)
	# 是不是下面这行有问题啊，epochs应该是轮数，下面这行重新定义了epochs，把下面的epochs改成epochs1试试看
	epochs1 = range(len(history.history['loss']))

	plt.figure(figsize=(15, 7))
	plt.plot(epochs1, history.history['accuracy'], 'b', label='Training acc')
	plt.plot(epochs1, history.history['val_accuracy'], 'r', label='Validation acc')
	plt.title('Traing and Validation accuracy', fontsize=16)
	plt.xlabel('Training Epoch', fontsize=16)
	plt.ylabel('Accuracy', fontsize=16)
	plt.legend()
	plt.savefig(f'D:/期刊论文薛泽昌/journal paper代码/QC桥梁深度学习/单个cable衰减/xue{n_filters}.jpg')

	plt.figure(figsize=(15, 7))
	plt.plot(epochs1, history.history['loss'], 'b', label='Training loss')
	plt.plot(epochs1, history.history['val_loss'], 'r', label='Validation val_loss')
	plt.title('Traing and Validation loss')
	plt.xlabel('Training Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(f'D:/期刊论文薛泽昌/journal paper代码/QC桥梁深度学习/单个cable衰减/xue1{n_filters}.jpg')
	# plt.figure(figsize=(15, 7))
	# plt.plot(history.history['accuracy'], "g--", label="Accuracy of training data")
	# plt.plot(history.history['val_accuracy'], "g", label="Accuracy of validation data")
	# plt.plot(history.history['loss'], "r--", label="Loss of training data")
	# plt.plot(history.history['val_loss'], "r", label="Loss of validation data")
	# plt.title('Model Accuracy and Loss')
	# plt.ylabel('Accuracy and Loss')
	# plt.xlabel('Training Epoch')
	# plt.ylim(0)
	# plt.legend()
	# plt.show()
	# 把下面这行移到这里就不报错了，如果_,accuracy放到fit前面就会报错，too many 什么玩意的。还是报错，是在算完所有epoch后又跳出来too many values to unpack，不影响计算。如果这行放前面压根算不了
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return accuracy

# 定义混淆矩阵
def plot_confusion_matrix(cm,classes,title='Confusion matrix',cmap=plt.cm.jet):#cmap是用于设置混淆矩阵的颜色和主题
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	thresh = cm.max() / 2.
	for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, '{:.2f}'.format(cm[i,j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.savefig(r'D:/期刊论文薛泽昌/journal paper代码/QC桥梁深度学习/单个cable衰减/confusion.jpg')
	#plt.show()


# 显示混淆矩阵
def plot_confuse(model, x_val, y_val):
	predictions = model.predict_classes(x_val)
	truelabel = y_val.argmax(axis=-1)  # 将one-hot转化为label
	conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
	plt.figure(figsize=(9, 7))
	plot_confusion_matrix(conf_mat, range(np.max(truelabel) + 1))

def summarize_results1(scores, params):#和summarize_results不同的是该函数考虑了多个params条件下的结果展示
	print(scores, params)
	# summarize mean and standard deviation
	for i in range(len(scores)):
		m, s = mean(scores[i]), std(scores[i])
		print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))
	# boxplot of scores
	pyplot.boxplot(scores, labels=params)
	pyplot.savefig('exp_cnn_filters.png')


# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

def run_experiment1(params,repeats=1):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	allscores = list()
	for p in params:
		scores = list()
		for r in range(repeats):
			score = evaluate_model(trainX, trainy, testX, testy, p)
			score = score * 100.0
			print('>p=%d #%d: %.3f' % (p, r+1, score))
			scores.append(score)
			allscores.append(scores)
	# summarize results
	summarize_results1(allscores,params)

# run an experiment
def run_experiment(repeats=1):
	# load data
	trainX, trainy, testX, testy = load_dataset()
	# repeat experiment
	scores = list()
	for r in range(repeats):
		score = evaluate_model(trainX, trainy, testX, testy)
		score = score * 100.0
		print('>#%d: %.3f' % (r + 1, score))
		scores.append(score)
	# summarize results
	summarize_results(scores)

n_params = [0.25,0.45,0.6,0.75]
# run the experiment
run_experiment1(n_params)

