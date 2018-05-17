# from __future__ import division

import sys
import re
import string
import re
import collections

file = open(sys.argv[1], 'r')

# file = open('sample.txt', 'r')

training_file = file.read()

training_file = training_file.strip()
training_file = re.sub('[!#"%$&)(+*-,/.;:=<?>@\[\]~]',' ',training_file.lower())
training_file = training_file.splitlines()

stopwords = set(["a","about","above","after","again","all","am","an","and","any","are","aren't","as","at","be","because","been","i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","can","will","just","don","should","now","before","being","below","between","both","but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't","doing","don't","down","during","each","few","for","from","further","had","hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her","here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only","or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's","the","their","theirs","them","themselves","then","there","there's","these","they","they'd","they'll","they're","they've","this","those","through","to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were","weren't","what","what's","when","when's","where","where's","which","while","who","who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll","you're","you've"])
weights_true = collections.defaultdict(int)
weights_pos = collections.defaultdict(int)
training_instance = collections.defaultdict(lambda : '')
word_count = collections.defaultdict(lambda : '')
bias_true = 0
bias_pos = 0
y_true=0
y_pos=0
counter = 1
cached_weights_true = 0
cached_bias_true = 0
cached_weights_pos = 0
cached_bias_pos = 0

for line in training_file:

	words = line.strip().split(" ")
	# training_instance[words[0]] = words[1:]
	for word in words[3:]:
		if word in word_count:
			word_count[word] += 1
		else:
			word_count[word] = 1

# print training_instance
# exit()
keys = sorted(word_count,key=word_count.get,reverse=True)

for i in range(10):
	stopwords.add(keys[i])

file_stop = open("stopwords.txt",'w')
for word in stopwords:
	file_stop.write(word+"\n")
file_stop.close()

MAX_ITER = 15

for iter in range(MAX_ITER):
	for line in training_file:
		# print training_instance[key]
		words = line.strip().split(" ")
		feature_count = collections.defaultdict(int)
		temp_word_list = []
		y_pos = 1 if words[2] == "pos" else -1
		y_true = 1 if words[1] == "true" else -1
		for word in words[3:]:
			if word not in stopwords:
				if word in feature_count:
					feature_count[word] += 1
				else:
					feature_count[word] = 1
				# temp_word_list.append(word)
		# print feature_count
		# exit()

		activation = 0
		for word in feature_count:
			activation += feature_count[word]*weights_pos[word]
		activation += bias_pos
		if y_pos*(activation) <= 0:
			for word in feature_count:
				weights_pos[word] += feature_count[word]*y_pos
				cached_weights_pos += y_pos*counter*feature_count[word]
			bias_pos += y_pos
			cached_bias_pos += y_pos*counter

		activation = 0
		for word in feature_count:
			activation += feature_count[word]*weights_true[word]
		activation += bias_true	
		if y_true*(activation) <= 0:
			for word in feature_count:
				weights_true[word] += feature_count[word]*y_true
				cached_weights_true += y_true*counter*feature_count[word]
			bias_true += y_true
			cached_bias_true += y_true*counter
		counter += 1

out = open('vanillamodel.txt','w')
out2 = open('averagedmodel.txt','w')
out.write(str(bias_true) + ' ' + str(bias_pos) + ' ' + str(len(weights_true)) + ' ' + str(len(weights_pos)) + '\n')
out2.write(str(bias_true-(cached_bias_true)*(1/counter)) + ' ' + str(bias_pos-(cached_bias_pos)*(1/counter)) + ' ' + str(len(weights_true)) + ' ' + str(len(weights_pos)) + '\n')
for weight in weights_true:
	out.write(weight + ' ' + str(weights_true[weight]) + '\n')
	out2.write(weight + ' ' + str(weights_true[weight]-(cached_weights_true)*(1/counter)) + '\n')
for weight in weights_pos:
	out.write(weight + ' ' + str(weights_pos[weight]) + '\n')
	out2.write(weight + ' ' + str(weights_pos[weight]-(cached_weights_pos)*(1/counter)) + '\n')
out.close()
out2.close()
