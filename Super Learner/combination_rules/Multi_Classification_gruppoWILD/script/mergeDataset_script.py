from os import listdir
from os.path import isfile, join



onlyfiles = [f for f in listdir("./temp") if isfile(join("./temp", f))]


fout=open("./CSVs/CICIDS2017.csv","a")

    
for num in range(0,len(onlyfiles)):
		print(onlyfiles[num])
		f = open("./temp/"+onlyfiles[num],'r',encoding='latin-1')
		#if num!=0:
		next(f)
		for line in f:
			fout.write(line)
		
		f.close()
fout.close()

