import csv
import json
import sys

count = {'BENIGN' : 0,'Bot': 0,'DDoS' : 0,'DoSGoldenEye': 0,'DoSHulk': 0,'DoSSlowhttptest':  0,'DoSslowloris' :0,'FTP-Patator' :0,'Heartbleed':  0,'Infiltration': 0,'PortScan':  0,'SSH-Patator': 0,'WebAttackBruteForce': 0,'WebAttackSqlInjection':  0,'WebAttackXSS':  0}
#with open('./CSVs/Monday-WorkingHours.pcap_ISCX.csv', 'r') as inp:
with open(sys.argv[1], 'r') as inp:
	for row in csv.reader(inp):
		if any(row):
			count[row[len(row)-1]] +=1


print(count)
fout = open("./output/count_label.txt","w")
fout.write(json.dumps(count))
fout.close()

			
