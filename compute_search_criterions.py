import json, os, time

UNUSED_KEYS = ["logType","service","path","method","httpStatus",\
				"status","contentLength","responseTime","tags","fields",\
				"messages","query","header_user-agent",\
				"header_x-forwarded-for","header_x-forwarded-port","header_x-forwarded-proto",\
				"start","nbResults","epcMinRange","orderBy","reference","sort","region"]

USEFUL_KEYS = ["header_x-gu","propertyType","transferType",\
				"maxPrice","postalCode","city","countryCode","nbBedroom"]

#this function adds the values of the jsonObject associated to the keys in keyList into the given csv file
def appendToCSV(file,keyList,jsonObject,geocodings):
	if "minPrice" in jsonObject and jsonObject["countryCode"] == ['BE'] and jsonObject["maxPrice"] > jsonObject["minPrice"] \
	and 2000000 > jsonObject["maxPrice"] and ((jsonObject["transferType"] == ['SALE'] and jsonObject["maxPrice"] > 30000)\
	or jsonObject["transferType"] == ['RENT'] and jsonObject["maxPrice"] > 100 and jsonObject["maxPrice"] < 5000) and ("city" in jsonObject or "postalCode" in jsonObject)\
	and (jsonObject["propertyType"] == ['HOUSE'] or jsonObject["propertyType"] == ['APARTMENT']) \
	and ("nbBedroom" in jsonObject and 0 <= jsonObject["nbBedroom"] <= 10):
		try:
			if "city" in jsonObject:
				for i in range(0,len(jsonObject["city"])):
					if jsonObject["city"][i] in geocodings:
						for key in keyList:
							if key == "city":
								file.write(str(geocodings[jsonObject[key][i]]["lat"])+";"+str(geocodings[jsonObject[key][i]]["lon"])+";")
							elif key == "countryCode":
								continue
							elif key == "propertyType":
								if jsonObject["propertyType"] == ['HOUSE']:
									file.write(str(0)+";")
								elif jsonObject["propertyType"] == ['APARTMENT']:
									file.write(str(1)+";")
							elif key == "transferType":
								if jsonObject["transferType"] == ['SALE']:
									file.write(str(0)+";")
								elif jsonObject["transferType"] == ['RENT']:
									file.write(str(1)+";")
							elif key in jsonObject and key != "postalCode":
								file.write(str(jsonObject[key])+";")
						file.write("\n")
				
			if "postalCode" in jsonObject:
				for i in range(0,len(jsonObject["postalCode"])):
					if jsonObject["postalCode"][i] in geocodings:
						for key in keyList:
							if key == "postalCode":
								file.write(str(geocodings[jsonObject[key][i]]["lat"])+";"+str(geocodings[jsonObject[key][i]]["lon"])+";")
							elif key == "countryCode":
								continue
							elif key == "propertyType":
								if jsonObject["propertyType"] == ['HOUSE']:
									file.write(str(0)+";")
								elif jsonObject["propertyType"] == ['APARTMENT']:
									file.write(str(1)+";")
							elif key == "transferType":
								if jsonObject["transferType"] == ['SALE']:
									file.write(str(0)+";")
								elif jsonObject["transferType"] == ['RENT']:
									file.write(str(1)+";")
							elif key in jsonObject and key != "city":
								file.write(str(jsonObject[key])+";")
						file.write("\n")
		except KeyError as key:
			print(key)

#takes the attributes of the json object associated to key and adds them to the atttibute of jsonObject
def extractJsonObject(jsonObject, key):
	jsonObject.update(jsonObject[key])
	jsonObject.pop(key, None)

if __name__ == '__main__':
	#clean the existing files and open them to write in them
	open("searchLogs","w").close()
	open("searchCriterion.csv","w").close()
	cleanedFile = open("searchLogs","a",encoding='UTF')
	csvFile = open("searchCriterion.csv","a",encoding='UTF')

	geocodings_file = open("geocodings.csv","r",encoding='UTF')
	geocodings = {}
	for line in geocodings_file:
		line = line.split(";")
		geocodings[line[0]] = {'lat': float(line[2]), 'lon': float(line[3].replace("\n",""))}
		geocodings[line[1]] = {'lat': float(line[2]), 'lon': float(line[3].replace("\n",""))}
	geocodings_file.close()
	
	#list of the files located in the data folder
	logsFilesList = os.listdir("data-5_04__7_06")
	hashTable = {}
	start = time.time()
	for logsFile in logsFilesList:
		print("File",logsFile," is being handled(critere de recherche)..."," ||",'time:', time.time() - start, end="\r")
			#data-9_02__4_04
		originalFile = open("data-5_04__7_06\\"+logsFile,encoding='UTF')
		for line in originalFile:
			try:
				tmp = line.split("Z ")
				line = tmp[1]
				if "method" in line:
					jsonObject = json.loads(line)
					extractJsonObject(jsonObject,"body")
					extractJsonObject(jsonObject,"score")
					if jsonObject["method"] == "POST" and "search" in jsonObject["path"] and jsonObject["start"] == 0\
						and "propertyType" in jsonObject and "transferType" in jsonObject and "header_x-gu" in jsonObject and jsonObject["header_x-gu"] != "":
						for key in UNUSED_KEYS:
							jsonObject.pop(key, None)
						#jsonObject["header_x-forwarded-for"] = jsonObject["header_x-forwarded-for"].split(",")[0]
						key = ""
						for keys in USEFUL_KEYS:
							if keys in jsonObject:
								key += str(jsonObject[keys])
						if key not in hashTable:
							hashTable[key] = False
						if key in hashTable and hashTable[key] == False:
							hashTable[key] = True
							appendToCSV(csvFile,USEFUL_KEYS,jsonObject,geocodings)
							cleanedFile.write(str(jsonObject)+"\n")
			except KeyError:
				continue
			except ValueError:
				continue
			except IndexError:
				continue

		originalFile.close()
		
	cleanedFile.close()
	csvFile.close()