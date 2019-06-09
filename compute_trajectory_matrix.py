import json
import os
import time

UNUSED_KEYS = ["logType", "service", "method", "httpStatus",
               "status", "contentLength", "responseTime", "tags", "fields",
               "messages", "query", "header_user-agent",
               "header_x-forwarded-port", "header_x-forwarded-proto",
               "description", "name", "type", "origin", "eventType", "date", "header_x-forwarded-for" ]

USEFUL_KEYS = ["header_x-gu","path"]

UNUSED_VALUES = ["FINDDREAMHOUSE", "OURGOODS", "RENTAHOUSE", "SELLAHOUSE", "ESTIMATEAHOUSE", "REALESTATEDEVELOPER", "LUXAUTO", "Type", "alle", "Continuer", "Doorgann", "Suivant", "PrÃ©cÃ©dent", "Rechercher", "Zoeken",
                 "Acheteur", "Koper", "Verhuren", "koper", "Louer", "Huurder", "huurder", "Vendre", "Verkopen", "verkoper", " investeerder", "eigenaar", "CTA-Voir tous nos biens", "CTA-Bekijk al onze panden", "CTA-Voir Le Bien", "Vignette"]
# takes the attributes of the json object associated to key and adds them to the atttibute of jsonObject


def extractJsonObject(jsonObject, key):
    jsonObject.update(jsonObject[key])
    jsonObject.pop(key, None)

if __name__ == "__main__": 
    # clean the existing files and open them to write in them
    open("cleanedLogs", "w").close()
    open("ipList", "w").close()
    cleanedFile = open("cleanedLogs", "a", encoding='UTF')
    file = open("ipList", "a", encoding='UTF')

    ipDict = {}

    logsFilesList = os.listdir("data")
    hashTable = {}
    start = time.time()
    for logsFile in logsFilesList:
        print("File", logsFile, " is being handled(cleanage en cours)...",
            " ||", 'time:', time.time() - start, end="\r")
        originalFile = open("data\\"+logsFile, encoding='UTF')
        for line in originalFile:
            try:
                tmp = line.split("Z ")
                line = tmp[1]
                if "method" in line:
                    jsonObject = json.loads(line)
                    extractJsonObject(jsonObject, "body")
                    # if (jsonObject["method"] == "POST" and (("monitoring" in jsonObject["path"] and
                    #                                          "value" in jsonObject and "image" in jsonObject["value"])or ("search" in jsonObject["path"] and jsonObject["start"] == 0
                    #                                                                                                       and "propertyType" in jsonObject and "transferType" in jsonObject)))\
                    if "header_x-gu" in jsonObject and jsonObject["header_x-gu"] != "" and jsonObject["method"] == "GET" and "v1/properties" in jsonObject["path"] and "ref" not in jsonObject["path"]:
                        for key in UNUSED_KEYS:
                            jsonObject.pop(key, None)
                        jsonObject["header_x-gu"] = jsonObject["header_x-gu"].split(",")[
                            0]
                        key = ""
                        for keys in USEFUL_KEYS:
                            if keys in jsonObject:
                                key += str(jsonObject[keys])
                        if key not in hashTable:
                            hashTable[key] = False
                        if key in hashTable and hashTable[key] == False:
                            hashTable[key] = True
                            ipDict[str(jsonObject["header_x-gu"])] = []
                            cleanedFile.write(json.dumps(jsonObject)+"\n")
            except KeyError:
                continue
            except ValueError:
                continue
            except IndexError:
                continue

        originalFile.close()
    for ip in ipDict:
        file.write(ip+"\n")
    file.close()
    cleanedFile.close()

    cleanedFile = open("cleanedLogs", "r", encoding='UTF')

    for line in cleanedFile:
        jsonObject = json.loads(line)
        ipDict[str(jsonObject["header_x-gu"])].append(jsonObject["path"].split("/")[4])

    nb_file = 1
    nb_line = 1

    open("stepByStep"+str(nb_file)+".csv", "w").close()
    stepFile = open("stepByStep"+str(nb_file)+".csv", "a", encoding='UTF')
    for ip in ipDict:
        print("Trajectories are being written...",
            " ||", 'time:', time.time() - start, end="\r")
        if nb_line > 20000000:
            stepFile.close()
            nb_file += 1
            nb_line = 1
            open("stepByStep"+str(nb_file)+".csv", "w").close()
            stepFile = open("stepByStep"+str(nb_file)+".csv", "a", encoding='UTF')
        if len(ipDict[ip]) > 3:
            stepFile.write(ip+";")        
            for i in range(len(ipDict[ip])):
                stepFile.write(str(ipDict[ip][i])+";")
            stepFile.write("\n")
            nb_line += 1
        
    cleanedFile.close()
    stepFile.close()
