from bs4 import BeautifulSoup as Soup
import re, nltk, os, unidecode, requests
import time


def get_geocodings(geocoding_file,url,site,translationDictionary,coord_type):

	response = requests.get(url)
	soup = Soup(response.text, features="html.parser")

	toDecompose = soup.find_all(["h2","h1"])
	for i in range(0,len(toDecompose)):
		toDecompose[i].decompose()

	pc_list = soup.find("div",class_="mw-parser-output").find_all("ul",recursive="",limit=13)

	start = time.time()
	count1 = 0
	for tmp in pc_list:
		sub_pc_list = tmp.find_all("li")
		count1 += 1
		count2 = 0
		for tmp2 in sub_pc_list:
			has_city = True
			count2 += 1
			print("pc_list :",count1,'/',len(pc_list)," || sub_pc_list :",count2,'/',len(sub_pc_list)," ||",',time:', time.time() - start, end="\r")
			if "\t" in tmp2.text:
				tmp_split = tmp2.text.split("\t")
			elif "\n" in tmp2.text:
				tmp_split = tmp2.text.split("\n")
				tmp_split = tmp_split[0].split(" ")
				if len(tmp_split) == 1:
					has_city = False
			else:
				tmp_split = tmp2.text.split(" ")
			if tmp_split[0].replace("\n","").isdigit():
				pc = tmp_split[0]

			if tmp2.a != None and has_city:
				city_name = tmp2.a.text
				page = tmp2.a["href"]
				try:
					response = requests.get(site+page)
					soup = Soup(response.text, features="html.parser")

					if coord_type == "fr" :
						if soup.span != None and soup.span.a != None:
							lat = soup.span.a["data-lat"]
							lon = soup.span.a["data-lon"]
							geocoding_file.write(city_name+";"+pc+";"+lat+";"+lon+"\n")
							if city_name in translationDictionary:
								geocoding_file.write(translationDictionary[city_name]+";"+pc+";"+lat+";"+lon+"\n")
						else:
							print(city_name)
				except (KeyError,requests.ConnectionError) as err:
					print(city_name)

if __name__ == '__main__':

	open("geocodings.csv","w").close()
	geocoding_file = open("geocodings.csv","a",encoding="UTF")

	site = "https://fr.wikipedia.org/wiki/Liste_des_communes_belges_et_leur_traduction"

	response = requests.get(site)
	soup = Soup(response.text, features="html.parser")

	tmp = soup.find_all("tbody")
	brusselRegion = tmp[0].find_all("td")
	flemishRegion = tmp[1].find_all("td")
	walloonRegion = tmp[2].find_all("td")

	dict = {}

	for i in range(0,len(brusselRegion),5):
		dict[brusselRegion[i].text.replace("\n","")] = brusselRegion[i+2].text.replace("\n","")

	for i in range(0,len(flemishRegion),7):
		if flemishRegion[i+4].text.replace("\n","") != "":
			dict[flemishRegion[i+4].text.replace("\n","")] = flemishRegion[i].text.replace("\n","")

	for i in range(0,len(walloonRegion),8):
		if walloonRegion[i+4].text.replace("\n","") != "":
			dict[walloonRegion[i].text.replace("\n","")] = walloonRegion[i+4].text.replace("\n","")


	url = "https://fr.wikipedia.org/wiki/Liste_des_codes_postaux_belges"
	site = "https://fr.wikipedia.org"
	get_geocodings(geocoding_file,url,site,dict,"fr")

	geocoding_file.close()
