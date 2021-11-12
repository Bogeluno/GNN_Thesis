# Load packages
from bs4 import BeautifulSoup
import requests
import html
import re
import folium
import pickle

# S-train
r_S = requests.get('https://da.wikipedia.org/wiki/S-togsstationer').text
soup_S = BeautifulSoup(r_S, 'html.parser')
stations_S = [re.findall(r'title="([^"]+)',str(x.findAll(['td'])[0]))[0] for x in soup_S.findAll('table',{"class":"wikitable sortable"})[0].findAll(['tr'])[1:]]

# Metro
r_M = requests.get('https://en.wikipedia.org/wiki/List_of_Copenhagen_Metro_stations').text
soup_M = BeautifulSoup(r_M, 'html.parser')
stations_M = [re.findall(r'title="([^"|\()]+)',str(x))[0] for x in soup_M.findAll('table')[1].findAll(['tr'])[1:] if 'S-train' not in re.findall(r'title="([^"]+)',str(x.findAll(['td'])[-1]))]

# Regional
stations_R = ['Tårnby Station']

# Merge
stations = stations_S + stations_M + stations_R

# Get location function
def get_location(s):
    r = requests.get(f"http://api.positionstack.com/v1/forward?access_key=6dc13a7ce41f830eb485af606a79617b&query={s}").json().get('data')

    r_dk = [l for l in r if l.get('country') == 'Denmark']

    if len(r_dk) == 0:
        return None
    else:
        return (r_dk[0].get('latitude'), r_dk[0].get('longitude'))

# Apply
stations_loc = {}
for s in stations:
    try:
        stations_loc[s] = get_location(s)
    except:
        pass


# Manual fixes/addtions
stations_loc['Husum Station'] = (55.709416513900315, 12.464032127468773)
stations_loc['Fasanvej Station'] = (55.68175346374325, 12.523123655665945)
stations_loc['Forum Station'] = (55.68194175606065, 12.552430497994727)
stations_loc['Øresund Station'] = (55.66135306295389, 12.628820660777878)
stations_loc['Tårnby Station'] = (55.629946132691515, 12.602094497992477)
stations_loc['Høvelte Trinbræt'] = (55.85114141925344, 12.388094282659448)
stations_loc['Sundby Station'] = (55.64525942491382, 12.585803299842478)
stations_loc['Rødovre Station'] = (55.6649407086372, 12.458544638473386)
stations_loc['Flintholm Station'] = (55.68582990624824, 12.499256667309936)
stations_loc['Islands Brygge Station'] = (55.66363850651178, 12.585191713884258)
stations_loc['Christianshavn Station'] = (55.672436293283894, 12.591146303186278)
stations_loc['Egedal Station'] = (55.77972548711175, 12.18539629799885)
stations_loc['Høje Taastrup Station'] = (55.64910485339314, 12.26935157749195)
stations_loc['Ishøj Station'] = (55.61335971792646, 12.357989482649314)
stations_loc['Herlev Station'] = (55.71897547452495, 12.443517888749625)
stations_loc['Bagsværd Station'] = (55.76164229643494, 12.45427312998459)
stations_loc['Virum Station'] = (55.79592019022332, 12.473300669163974)
stations_loc['Bella Center Station'] = (55.63814816923437, 12.582842567307864)
stations_loc['KB Hallen Station'] = (55.67776722523307, 12.492490326830145)
stations_loc['Frederiksberg Station'] = (55.681301715838, 12.531751584501528)
stations_loc['Birkerød Station'] = (55.61335971792646, 12.357989482649314)
stations_loc['Farum Station'] = (55.81214308083602, 12.373480884507142)
stations_loc['Ballerup Station'] = (55.72998289389706, 12.358148026832373)
stations_loc['Stenløse Station'] = (55.766402823281936, 12.190661542176398)
stations_loc['Ølstykke Station'] = (55.79571315450955, 12.15972272683513)
stations_loc['Danshøj Station'] = (55.664322692436514, 12.49376378450079)

# Save
with open('data/processed/Train_stations.pickle', 'wb') as handle:
    pickle.dump(stations_loc, handle, protocol=pickle.HIGHEST_PROTOCOL)