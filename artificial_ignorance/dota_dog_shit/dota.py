import time
import requests
import json
import numpy as np

from DotaDatabase import DotaDatabase

class Dota():
    # Class to load up all the dota data we will need

    def __init__(self, openDotaURL='https://api.opendota.com/api/'):
        self.baseURL = openDotaURL
        self.database = DotaDatabase()

        self.heroes = self.load_heroes()

    def getCollection(self, collection):
        return self.database.getAllRecords(collection=collection)

    # Hero functionality
    # Load heroes from database, if none exist, load from opendota
    def load_heroes(self):
        collection = self.getCollection('heroes')
        if len(collection) == 0:
            heroes = self.getHeroes()
            for hero in heroes:
                self.database.storeHero(hero)
            collection = self.getCollection('heroes')
        collection = np.array(collection)
        return collection

    # Get heroes from opendota
    def getHeroes(self):
        res = requests.get(f'{self.baseURL}/heroes')
        heroes = json.loads(res.content)
        return np.array(heroes)
    
    # Init the database
    def init_heroes_db(self):
        for hero in self.heroes:
            self.database.storeHero(hero)

    def getLeagues(self):
        res = requests.get(f'{self.baseURL}/leagues')
        return json.loads(res.content)
    
    def getLeague(self, leagueId):
        res = requests.get(f'{self.baseURL}/leagues/{leagueId}')
        return json.loads(res.content)
    
    def getLeagueMatches(self, leagueId):
        res = requests.get(f'{self.baseURL}/leagues/{leagueId}/matches')
        return json.loads(res.content)


    def getMatchDetails(self, matchId):
        res = requests.get(f'{self.baseURL}/matches/{matchId}')
        return json.loads(res.content)

    def getProMatches(self):
        res = requests.get(f'{self.baseURL}/proMatches')
        return json.loads(res.content)
    
    def loadProMatches(self):
        leagues = self.getLeagues()
        leagues = leagues[::-1]
        for league in leagues:
            print("loading matches for league: ", league['name'])
            matches = self.getLeagueMatches(league['leagueid'])
            matches = [match['match_id'] for match in matches]
            print("Match count: ", len(matches))
            for match in matches:
                time.sleep(2)
                print("Loading match: ", match)
                records = self.database.getRecord(query={'match_id': match}, collection='matches')
                if len(records) > 0:
                    print("Match already exists in database")
                    break
                else:
                    try:
                        match_details = self.getMatchDetails(match)
                        if match_details['start_time'] >= 1546378884:
                            self.database.storeMatch(match_details)
                        else:
                            print("Match too old, skipping")
                            break
                    except:
                        continue
            print("Current Matches size:", len(self.database.getAllRecords('matches')))
        


# d = Dota()
# d.loadProMatches()


# leagues = d.getLeagues()
# l = d.getLeague(leagues[-1]['leagueid'])
# print(l)
# matches = d.getLeagueMatches(leagues[-1]['leagueid'])
# print(matches)
# pro_ids = [l['leagueid'] for l in leagues if l['tier'] == 'professional']
# print(pro_ids)
