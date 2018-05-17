import sys
import re
import collections

test_file = open(sys.argv[2], 'r').read().splitlines()
stopwords = open('stopwords.txt','r').read().splitlines()
data = open(sys.argv[1],'r').read().splitlines()

first_line = data[0].split(' ')

bias_true = int(first_line[0])
bias_pos = int(first_line[1])

weightsTF = collections.defaultdict(int)
weightsPN = collections.defaultdict(int)

for i in range(1,int(first_line[2])+1):
	temp = data[i].split(' ')
	weightsTF[temp[0]] = int(temp[1])

for i in range(int(first_line[2])+1,int(first_line[2])+int(first_line[3])+1):
	temp = data[i].split(' ')
	weightsPN[temp[0]] = int(temp[1])

file = open("percepoutput.txt",'w')

for line in test_file:
	line = re.sub('[!#"%$&)(+*-,/.;:=<?>@\[\]~]',' ',line)
	words = line.split(" ")
	file.write(words[0]+" ")
	feature_count = collections.defaultdict(int)
	temp_word_list = []
	for word in words[1:]:
		word = word.lower()
		if word not in stopwords:
			if word in feature_count:
				feature_count[word] += 1
			else:
				feature_count[word] = 1
		# temp_word_list.append(word)
	activation = 0
	for word in feature_count:
		activation += feature_count[word]*weightsTF[word]
	activation += bias_true
	if  activation > 0:
		file.write("True ")
	else:
		file.write("Fake ")
	activation = 0
	for word in feature_count:
		activation += feature_count[word]*weightsPN[word]
	activation += bias_pos
	if activation > 0:
		file.write("Pos\n")
	else:
		file.write("Neg\n")

file.close()







# word_count = dict()
# probs = probs[1:]
# for line in probs:
# 	line = line.split(' ')
# 	if line[0] in word_count:
# 		word_count[line[0]][line[1]] = float(line[2])
# 	else:
# 		word_count[line[0]] = dict()
# 		word_count[line[0]][line[1]] = float(line[2])

# for line in test_file:
# 	pos_prob = prior_probability['pos']
# 	neg_prob = prior_probability['neg']
# 	true_prob = prior_probability['true']
# 	fake_prob = prior_probability['fake']

# 	line = re.sub('[!#"%$&)(+*-,/.;:=<?>@\[\]~]',' ',line)
# 	out1.write(line.split(' ')[0] + ' ')
# 	words = line.split(' ')[1:]
	

# 	for word in words:
# 		word = word.lower()
# 		if (word not in stopwords) and (word in uniqueWords):
# 			pos_prob += word_count[word]['pos']
# 			neg_prob += word_count[word]['neg']
# 			true_prob += word_count[word]['true']
# 			fake_prob += word_count[word]['fake']

# 	if max(true_prob,fake_prob) == true_prob:
# 		out1.write("True ")
# 	else:
# 		out1.write("Fake ")

# 	if max(pos_prob,neg_prob) == pos_prob:
# 		out1.write("Pos" + '\n')
# 	else:
# 		out1.write("Neg" + '\n')
