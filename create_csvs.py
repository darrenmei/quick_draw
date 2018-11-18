import numpy as np
import pandas as pd
import sys
import csv

# Cutoffs derived from quantile cuts from master tsv

def extractPDs(inputfiles, train, dev, test):
	"""
    	Creates train.csv, dev.csv, test.csv
      	Arguments:
    	inputfile -- raw data tsv file
    	train -- file name
    	dev -- file name
    	test -- file name
    	Returns:
    	nothing
    	"""
	list = []
	sizes = []
	for file in inputfiles:
		print('file: ', file)
		df_temp = pd.read_csv(file, sep = ',', header = 0)
		sizes.append(df_temp.shape[0])

	max_len = min(sizes)   # Make sure we have uniform catetegory sizes
	count = 0
	for file in inputfiles:
		if count == 0:
			df_temp = pd.read_csv(file, sep = ',', header = 0)
			list.append(df_temp.loc[0:max_len, :])
		else:
			df_temp = pd.read_csv(file, sep = ',', header = 0)
			list.append(df_temp.loc[1:max_len, :])
	df = pd.concat(list)
	data = df[['drawing', 'key_id', 'word']]
	#print('data: ', data)

	masterLength = len(data['drawing'])
	# Shuffle data
	data = data.sample(frac=1, random_state=1).reset_index(drop=True)
	#print('suffled data: ', data)
	#data = data.sort_values()
	#print('suffled data: ', data)

	# Need to define cutoffs
	traindf = data.sort_index().truncate(before = 0, after = 364983) # 60%
	devdf = data.sort_index().truncate(before = 364984, after = 486644) # 20%
	testdf = data.sort_index().truncate(before = 486644, after = 608305) # 20%
	print ("Size of training dataframe is " + str(traindf.shape))
	print ("Size of dev dataframe is " + str(devdf.shape))
	print ("Size of test dataframe is " + str(testdf.shape))
	print ("Size of master dataframe is " + str(df.shape))
	traindf.to_csv(train, index = False)
	devdf.to_csv(dev, index = False)
	testdf.to_csv(test, index = False)


def main():
	np.random.seed(seed=1)
	if len(sys.argv) != 9:
		raise Exception("usage: python extractPDs.py <infiles>.tsv <train>.csv <dev>.tsv <test>.tsv")
	airplane, campfire, key, palm_tree, moon, trainCSV, devCSV, testCSV = sys.argv[1:10]
	list = []
	list.append(airplane)
	list.append(campfire)
	list.append(key)
	list.append(palm_tree)
	list.append(moon)
	extractPDs(list, trainCSV, devCSV, testCSV)

if __name__ == '__main__':
	main()
