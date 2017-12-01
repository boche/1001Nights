from aligner import *
import operator

file = open('decay24_test', 'r')
counter = 0
gt_drop = {}
greedy_drop = {}
id = 0
for line in file:
	id += 1
	print id
	if line.startswith('<Ground Truth>'):
		gt = line.split(': ')[1]
		if len(gt) <= 1: 
			continue
		else:
			gt = gt.split(' ')
	elif line.startswith('<Greedy>'):
		counter += 1
		if counter == 3000: 
			break
		greedy = line.split(': ')[1]
		if len(greedy) <= 1:
			continue
		else:
			greedy = greedy.split(' ')
		alignments = align(gt, greedy)
		idx = [row[0] for row in alignments[0]]
		for i in range(len(gt)):
			if (i + 1) not in idx:
				if gt[i] in gt_drop:
					gt_drop[gt[i]] += 1
				else:
					gt_drop[gt[i]] = 1
		idx = [row[1] for row in alignments[0]]			
		for i in range(len(greedy)):
			if (i + 1) not in idx:
				if greedy[i] in greedy_drop:
					greedy_drop[greedy[i]] += 1
				else:
					greedy_drop[greedy[i]] = 1
gt_drop = sorted(gt_drop.items(), key=operator.itemgetter(1), reverse=True)
greedy_drop = sorted(greedy_drop.items(), key=operator.itemgetter(1), reverse=True)
print 'ground truth drop: ', gt_drop[:50]
print 'greedy drop: ', greedy_drop[:50]



