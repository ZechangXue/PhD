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
#from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
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
import seaborn as sns
from keras.regularizers import l1_l2


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
	filenames = filenames1 + filenames2 + filenames3 + filenames4 + filenames5 + filenames6 + filenames7 + filenames8 + filenames9
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
	print(trainy)
	print(trainX.shape, trainy.shape)
	# load all test
	trainX, testX, trainy, testy = train_test_split(trainX,trainy,test_size=0.2,random_state=0)
	print(testX.shape, testy.shape)
	# zero-offset class values
	trainy = trainy - 1
	testy = testy - 1

	trainy = to_categorical(trainy)
	testy = to_categorical(testy)
	print(trainX.shape, trainy.shape, testX.shape, testy.shape)

	print(testy)
	print(trainy)
	return trainX, trainy, testX, testy
def evaluate_model(trainX, trainy, testX, testy, kernel_size, filters, Batch_size):

	verbose, epochs, batch_size = 1, 300, Batch_size
	n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
	# 创建L1/L2正则化对象
	#reg = l1_l2(l1=0.001, l2=0.001)
	model = Sequential()
	model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu',
					 input_shape=(n_timesteps, n_features)))
	model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))

	model.add(BatchNormalization(axis=2))
	model.add(MaxPooling1D(pool_size=3))
	model.add(Flatten())
	# dropout（0.5）就是每次迭代让一半卷积核不更新，这样可以防止过拟合现象
	model.add(Dropout(0.65))

	# flatten是一个降维过程（默认按行降维），如a=([[1,2],[3,4],[5,6]])，a.flatten()就=([1,2,3,4,5,6])
	# flatten('F')是按列降维，flatten('A')是按行降维，则a.flatten('F')=([1,3,5,2,4,6])

	# 下式是全连接层的输入层，dense就是全连接层，100代表着unit，也就是全连接层有100个元素组成
	model.add(Dense(300, activation='relu'))
	# 下式是全连接层的输出层，n_outputs值是6（分为六类标签），就是说该输出层由六个元素组成，再经过softmax转化成每种分类的概率
	model.add(Dense(n_outputs, activation='softmax'))
	plot_model(model, show_shapes=True, to_file='multichannel.png')
	# 用adam
	adam = tf.keras.optimizers.Adam(lr=0.00035, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0)
	#lr_metric = get_lr_metric(adam)
	#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', lr_metric])
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	model.get_weights()#查看权重

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
	plt.savefig(r'D:/期刊论文薛泽昌/journal paper代码/QC桥梁深度学习/单个cable衰减/xue.jpg')

	plt.figure(figsize=(15, 7))
	plt.plot(epochs1, history.history['loss'], 'b', label='Training loss')
	plt.plot(epochs1, history.history['val_loss'], 'r', label='Validation val_loss')
	plt.title('Traing and Validation loss')
	plt.xlabel('Training Epoch')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig(r'D:/期刊论文薛泽昌/journal paper代码/QC桥梁深度学习/单个cable衰减/xue1.jpg')
	_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
	return model, history

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
	classes = model.predict(x_val)
	predictions = np.argmax(classes, axis=1)
	truelabel = y_val.argmax(axis=-1)#将one-hot转化为label
	conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
	plt.figure(figsize=(9, 7))
	plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))
	plot_accuracy_bar_from_confusion(conf_mat)


def plot_accuracy_bar_from_confusion(conf_mat):
	diagonal = conf_mat.diagonal()
	total_per_class = conf_mat.sum(axis=1)
	accuracies = diagonal / total_per_class

	plt.figure(figsize=(12, 7))

	# 使用Seaborn的颜色条
	color_palette = sns.color_palette("YlOrRd", 100)  # 为100%创建一个颜色条
	colors = [color_palette[int(a)] for a in accuracies]  # 根据每个state的准确率分配颜色

	sns.barplot(x=np.arange(len(accuracies)), y=accuracies, palette=colors)

	# 设置x轴的标签
	states = ["state" + str(i + 1) for i in range(len(accuracies))]
	plt.xticks(np.arange(len(accuracies)), states, rotation=45)

	plt.ylabel('Accuracy %', fontsize=14)
	plt.xlabel('State', fontsize=14)
	plt.title('Accuracy for each state based on confusion matrix', fontsize=16)
	plt.xticks(fontsize=12)  # x轴刻度
	plt.yticks(fontsize=12)  # y轴刻度

	# 添加颜色条来表示准确率的颜色对应关系
	norm = plt.Normalize(accuracies.min(), accuracies.max())
	sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
	sm.set_array([])
	plt.colorbar(sm, orientation="vertical", label='Accuracy %')

	plt.tight_layout()
	plt.savefig(r'D:/期刊论文薛泽昌/journal paper代码/QC桥梁深度学习/单个cable衰减/accuracy_per_state.jpg')

# summarize scores
def summarize_results(scores):
	print(scores)
	m, s = mean(scores), std(scores)
	print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


def plot_metric(histories, Batch_size_list, metric, title, file_name):
    plt.figure(figsize=(15, 7))
    for i, history_list in enumerate(histories):
        for history in history_list:
            plt.plot(history[metric], label=f'{metric.title()} (Filters={Batch_size_list[i]})')
    plt.title(title, fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(metric.title(), fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

def plot_all_curves(histories, Batch_size_list):
	all_data = []
	for metric in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
		plt.figure(figsize=(15, 7))
		for i, history_list in enumerate(histories):
			# Calculate average for each epoch across all repeats
			avg_metric = np.mean([h[metric] for h in history_list], axis=0)
			plt.plot(avg_metric, label=f'{metric.title()} (Batch_size={Batch_size_list[i]})')

			# Collect data for Excel
			for epoch, value in enumerate(avg_metric, start=1):
				all_data.append({
					'Batch Size': Batch_size_list[i],
					'Metric': metric,
					'Epoch': epoch,
					'Value': value
				})

		plt.title(f'Training and Validation {metric.title()}', fontsize=16)
		plt.xlabel('Epoch', fontsize=14)
		plt.ylabel(metric.title(), fontsize=14)
		plt.legend(fontsize=14)
		plt.tight_layout()
		plt.savefig(f'{metric}_plot.jpg')
		plt.close()

	# Create DataFrame from collected data
	df = pd.DataFrame(all_data)
	# Save to Excel
	df.to_excel("all_metrics_summary.xlsx", index=False)



# def plot_all_curves(histories, Batch_size_list):
# 	for metric in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
# 		plt.figure(figsize=(15, 7))
# 		for i, history_list in enumerate(histories):
# 			# Calculate average for each epoch across all repeats
# 			avg_metric = np.mean([h[metric] for h in history_list], axis=0)
# 			plt.plot(avg_metric, label=f'{metric.title()} (Batch_size={Batch_size_list[i]})')
#
# 		plt.title(f'Training and Validation {metric.title()}', fontsize=16)
# 		plt.xlabel('Epoch', fontsize=14)
# 		plt.ylabel(metric.title(), fontsize=14)
# 		plt.legend(fontsize=14)
# 		plt.tight_layout()
# 		plt.savefig(f'{metric}_plot.jpg')
# 		plt.close()

def run_experiment(repeats=3):
    # Define different filters
    Batch_size_list = [128, 256, 512]
    all_histories = []

    for Batch_size in Batch_size_list:
        # Load data
        trainX, trainy, testX, testy = load_dataset()
        # Repeat experiment
        histories = []
        for r in range(repeats):
            _, history = evaluate_model(trainX, trainy, testX, testy, 11, 7, Batch_size)  # kernel_size fixed at 11
            histories.append(history.history)
        all_histories.append(histories)

    # Plot all averaged curves
    plot_all_curves(all_histories, Batch_size_list)


# run the experiment
run_experiment()

