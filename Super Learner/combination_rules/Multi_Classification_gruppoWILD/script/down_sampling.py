from os import listdir
from os.path import isfile, join
import math
import csv
import numpy  as np

onlyfiles = [f for f in listdir("./CSVs") if isfile(join("./CSVs", f))]



for num in range(0,len(onlyfiles)):
	count=0		
	count_el=0
	print(onlyfiles[num])
	with open("./CSVs/"+onlyfiles[num], 'r',encoding='latin-1') as inp, open('./temp/temp_'+onlyfiles[num] , 'w') as out_temp,open('./temp/final_data_'+onlyfiles[num] , 'w') as out_final :	
		writer_temp = csv.writer(out_temp)
		writer_final = csv.writer(out_final)
		for row in csv.reader(inp):
			if any(row):
				if row[len(row)-1]=="BENIGN":
					count +=1
					writer_temp.writerow(row)
				else:
					count_el +=1
					writer_final.writerow(row)

		#print("count : " , count)
		#print("count_el : " , count_el)
		elem=math.floor(count/10)
		
	with open("./temp/temp_" +onlyfiles[num], 'r') as inp_temp,open('./temp/final_data_'+onlyfiles[num] , 'a') as out_final :
		writer_t = csv.writer(out_final)
		index=np.random.choice(count,elem, replace=False)
		sorted(index, key=int)
		#print(len(index))

		reader=csv.reader(inp_temp)
		content=list(reader)
		for i in index:
			writer_t.writerow(content[i])

			
