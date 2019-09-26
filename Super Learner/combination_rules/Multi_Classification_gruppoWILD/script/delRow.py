import csv

with open('./CSVs/CICIDS2017.csv', 'r') as inp, open('./CSVs/CICIDS2017_edit.csv', 'w') as out:
	writer = csv.writer(out)
	for row in csv.reader(inp):
		if any(row):
			if row[5] != '0' and int(row[7]) > 10:
				row[84]=row[84].replace(" ","")
				row[84]=row[84].replace("","") #carattere strano presente nelle 
				row[84]=row[84].replace("Â","") 
				writer.writerow(row)

		
