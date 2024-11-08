from pymongo import MongoClient
from typing import Dict, Any

class DotaDatabase():

    def __init__(self, URI="mongodb://localhost:27017/"):
        try:
            self.client = MongoClient(URI)
            self.db = self.client['ragelock']

        except Exception as e:
            print(f'Error connecting to mongodb: {e}')

    def getCollectionCount(self, collection:str = None):
        return self.db[collection].count_documents({})
    
    def storeHero(self, hero:Dict[str, Any]):
        res = None
        try:
            res = self.db['heroes'].insert_one(hero)
        except Exception as e:
            print(f'Error storing hero: {e}')
        return res

    def storeMatch(self, match:Dict[str, Any]):
        res = None
        try:
            res = self.db['matches'].insert_one(match)
        except Exception as e:
            print(f'Error storing hero: {e}')
        return res

    def storeRecord(self, record:Dict[Any, Any], db:str = None, collection:str = None):
        res = None
        try:
            if db == None and collection == None:
                res = self.collection.insert_one(record)
            else:
                res = self.client[db][collection].insert_one(record)
        except Exception as e:
                print(f'Exception adding record to collection: {e}')
        return res

    def find_one(self, collection:str = None):
        return self.db[collection].find_one()

    def getAllRecords(self, collection:str = None):
        records = []
        for doc in self.db[collection].find():
            records.append(doc)
        return records

    def getRecord(self, query, collection:str = None):
        records = []
        for doc in self.db[collection].find(query):
            records.append(doc)
        return records
    
    def updateRecord(
        self, 
        oldRecord:Dict[Any, Any], 
        newRecord:Dict[Any, Any], 
        db:str = None, 
        collection:str = None
    ):
        res = ''
        newRecord = {"$set" : newRecord}
        if db == None and collection == None:
            res = self.collection.update_one(oldRecord, newRecord)
        else:
            res = self.client[db][collection].update_one(oldRecord, newRecord)
        return res

    def kill(self):
        self.collection.drop()
