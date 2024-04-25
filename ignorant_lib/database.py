from pymongo import MongoClient
from typing import Dict, Any

class Database():

    def __init__(self, database='test', URI="mongodb://localhost:27017/"):
        try:
            self.client = MongoClient(URI)
            self.db = self.client[database]

        except Exception as e:
            print(f'Error connecting to mongodb: {e}')

    def getCollectionCount(self, collection:str = None):
        return self.db[collection].count_documents({})
    
    def storeRecord(self, record:Dict[Any, Any], collection:str = None):
        res = None
        try:
            res = self.db[collection].insert_one(record)
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
        self.db.drop()
