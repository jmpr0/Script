import csv
import json

count = []
with open('./CSVs/CICIDS2017_corrected.csv', 'r') as inp:
	for row in csv.reader(inp):
		if any(row):
			for i in range(0, len(row)):
				if row[i]=="NaN" or row[i]=="Infinity":
					  count[0] +=1

print(count)


