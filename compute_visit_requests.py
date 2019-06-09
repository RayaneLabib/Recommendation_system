import json
import os
import data_processing as dp
UNUSED_KEYS = ["logType","service","method","httpStatus",\
				"status","contentLength","responseTime","tags","fields",\
				"messages","query","header_user-agent",\
				"header_x-forwarded-port","header_x-forwarded-proto",\
				"description","name","type","origin","eventType",]

USEFUL_KEYS = ["header_x-forwarded-for","date","value","gu"]

def appendToCSV(file,keyList,jsonObject):
	if "value" in keyList and (jsonObject["value"] == "CTA-DemanderVisiteBas" or jsonObject["value"] == "CTA-DemanderVisiteHaut"):
		for key in keyList:
			if key in jsonObject:
				file.write(str(jsonObject[key])+";")
			else:
				if key == "nbBedroom":
					file.write(";")
				else:
					file.write(";")
		file.write("\n")
		
#clean the existing files and open them to write in them
open("demandeVisite.csv","w").close()
file = open("demandeVisite.csv","a",encoding='UTF')

logsFilesList = os.listdir("data")
logsFilesList2 = ["i-07ccd74a8ebc8c3e6-000000"]
hashTable = {}
for logsFile in logsFilesList:
	print("File",logsFile," is being handled(cleanage loading)...")
	originalFile = open("data\\"+logsFile,encoding='UTF')
	for line in originalFile:
		try:
			tmp = line.split("Z ")
			line = tmp[1]
			if "method" in line:
				jsonObject = json.loads(line)
				if jsonObject["method"] == "POST" and (("monitoring" in jsonObject["path"] and\
				"value"  in jsonObject) or ("search" in jsonObject["path"] and jsonObject["start"] == 0\
				and "propertyType" in jsonObject and "transferType" in jsonObject)):
					for key in UNUSED_KEYS:
						jsonObject.pop(key, None)
					jsonObject["header_x-forwarded-for"] = jsonObject["header_x-forwarded-for"].split(",")[0]
					key = ""
					for keys in USEFUL_KEYS:
						if keys in jsonObject:
							key += str(jsonObject[keys])
					if key not in hashTable:
						hashTable[key] = False
					if key in hashTable and hashTable[key] == False:
						hashTable[key] = True
						appendToCSV(file,USEFUL_KEYS,jsonObject)
		except KeyError:
			continue
		except ValueError:
			continue
		except IndexError:
			continue

	originalFile.close()

file.close()