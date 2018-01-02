import csv

trainFile = 'data/train.csv'
# read train data from csv file
trainSet = []
validationSet = []
featureNum = 9

with open(trainFile, 'r', encoding='big5') as csvFile:
    trainReader = csv.reader(csvFile)
    for line in trainReader:
        #print(', '.join(line))
        trainSet.append(line)

for row in trainSet[1:featureNum+1]:
    print(row)