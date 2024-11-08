# from DotaDatabase import DotaDatabase
# from dota import Dota
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ignorant_lib.database import Database
from generator import Generator

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import models, layers, datasets, optimizers
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler

def main():
    print("Starting ragelock")
    r = Ragelock()
    print(r.model.predict(np.array([[0, 10, 2, 1000, 10, 200, 0, 100]])))
    # print(r.db.getAllRecords("matches"))

class Model():
    def __init__(self, model):
        self.model = model
    
    def predict(self, data):
        return self.decodeWinLose(self.model.predict(data))

    def save(self, path):
        self.model.save(path)

    def decodeWinLose(self, outcome):
        outcome = outcome[-1]
        return "Rage" if outcome == 1.0 else "Normal"


class Ragelock():
    # Did someone ragequit your game?
    def __init__(self):
        self.db = Database("ragelock_1")
        self.matches = self.db.getAllRecords("matches")

        # self.matches = [self.matches[0]]
        current_len = len(self.matches)
        self.generator = Generator(match_id=current_len + 1)

        if self.db.getAllRecords("matches") == []:
            self.populate_db()
        # self.populate_db()

        try:
            self.training_data, self.training_data_objects = self.load_training_data()
            print("Training data loaded: ", self.training_data.shape)
            print(len(self.training_data_objects))
        except Exception as e:
            print("Error loading training data: ", e)
            self.training_data = None
            print("No training data found - building a new set.")
            self.training_data, self.training_data_objects = self.build_training_set()


        try:
            self.players_data = self.load_players_data()
        except:
            print("No players data found - building a new set.")
            self.players_data = self.build_players_data(self.training_data)

        # model_history = self.train_players(self.players_data)
        # self.plot_history(model_history.history)

        try:
            self.model = models.load_model('deadlock_player_model')
        except:
            print("No model found, training a new one")
            self.training_history = self.train_players(self.players_data)
            self.model = models.load_model('deadlock_player_model')
            
        self.model = Model(self.model)


        # self.players_data = self.load_players_data()
        # print("Players data loaded: ", self.players_data.shape)
        # self.train(self.training_data)



    def load_training_data(self):
        with open('../data/deadlock_matches.npy', 'rb') as f:
            numpy_data = np.load(f, allow_pickle=True)
        with open('../data/deadlock_matches.json', 'r') as f:
            json_data = json.load(f)
        return numpy_data, json_data
    
    def load_players_data(self):
        with open('../data/deadlock_players.npy', 'rb') as f:
            numpy_data = np.load(f, allow_pickle=True)
        return numpy_data
    
    def build_training_set(self):
        # Builds a training set from the matches in the database
        print("Building Training Set, count: ", len(self.matches))
        res = None
        res_objects = []
        count = 0
        for match in self.matches:
            match_team_data = self.convert_match_to_training_data(match)
            if res is None:
                res = match_team_data['numpy_array']
                res_objects = [match_team_data['json']]
            else:
                res = np.concatenate((res, match_team_data['numpy_array']), axis=0)
                res_objects += [match_team_data['json']]
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} matches")
        self.save_training_data(res, res_objects)
        return res, res_objects
    
    def convert_match_to_training_data(self, match):
        # Takes a match and returns a numpy array with amber and saphire objects
        amber_team, amber_team_objects = self.convert_team_to_training_data(match['amber_hand'], 'amber_hand')
        saphire_team, saphire_team_objects = self.convert_team_to_training_data(match['saphire_flame'], 'saphire_flame')

        winner = match['winner']
        amber_winner = 1 if winner == 'amber_hand' else 0
        saphire_winner = 1 if winner == 'saphire_flame' else 0

        numpy_data = {
            'amber_team': [amber_team],
            'saphire_team': [saphire_team],
            'amber_souls': match['amber_hand']['team_souls'],
            'saphire_souls': match['saphire_flame']['team_souls'],
            'amber_win': amber_winner,
            'saphire_win': saphire_winner,
            'duration': match['duration'],
            'rage_status': 1 if match['rage_status'] else 0
        }

        data = {
            'amber_team': amber_team_objects,
            'saphire_team': saphire_team_objects,
            'amber_souls': match['amber_hand']['team_souls'],
            'saphire_souls': match['saphire_flame']['team_souls'],
            'amber_win': amber_winner,
            'saphire_win': saphire_winner,
            'duration': match['duration'],
            'rage_status': 1 if match['rage_status'] else 0
        }

        df = pd.DataFrame(numpy_data)
        np_array = df.to_numpy()
        return {
            'json': data,
            'numpy_array': np_array,
            'raw_data': data
        }
    
    def save_training_data(self, data, data_objects):
        with open('../data/deadlock_matches.npy', 'wb') as f:
            np.save(f, data)
        with open('../data/deadlock_matches.json', 'w') as f:
            json.dump(data_objects, f)
    
    def save_players_data(self, players_data):
        with open('../data/deadlock_players.npy', 'wb') as f:
            np.save(f, players_data)
    
    def convert_team_to_training_data(self, team, team_name):
        # Takes a team and returns a numpy array with the team's data
        res = []
        res_objects = []
        for player in team['players']:
            data = {
                'player_hero': self.generator.get_hero_id(player['hero']),
                'player_kills': player['kills'],
                'player_deaths': player['deaths'],
                'player_assists': player['assists'],
                'player_total_souls': player['total_souls'],
                'player_souls_per_minute': player['souls_per_minute'],
                'player_damage': player['player_dmg'],
                'player_healing': player['healing'],
                'player_obj_damage': player['object_dmg'],
                'player_team': 1 if team_name == 'amber_hand' else 0,
                'player_rage': 1 if 'rage_' in player['steam_name'] else 0
            }
            np_array = np.array(list(data.values()))  # Convert dict values directly to numpy array
            res.append(np_array)
            res_objects.append(data)
        return res, res_objects

    def populate_db(self):
        print("Generating 10000 matches")
        match_count = 0
        rage_count = 0
        for i in range(10000):
            t = self.generator.generate_example_match()
            if self.generator.validate_match_data(t):
                match_count += 1
                if t["rage_status"]:
                    rage_count += 1
                self.db.storeRecord(t, "matches")
            if i % 1000 == 0:
                print(f"Generated {i} matches")
        print(f"Match count: {match_count}")
        print(f"Rage count: {rage_count}")

    def build_players_data(self, data):
        amber_teams = data[:, 0]
        saphire_teams = data[:, 1]
        all_teams = np.concatenate((amber_teams, saphire_teams), axis=0)
        print("one team: ", len(all_teams[0]))
        all_players = None
        count = 0
        for team in all_teams:
            for player in team:
                if all_players is None:
                    all_players = np.array([player])
                else:
                    all_players = np.concatenate((all_players, np.array([player])), axis=0)
                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count} players")
                    print("player: ", player)

        self.save_players_data(all_players)
        
        
        return 

    def train_players(self, data):
        print("Training players")
        print("Data shape: ", data.shape)
        print("Data: ", data[0])

        column_names = ["hero_id", "kills", "deaths", "assists", "total_souls", "souls_per_minute", "player_dmg", "healing", "obj_dmg", "team", "rage"]
        df_columns = ["kills", "deaths", "assists", "total_souls", "souls_per_minute", "player_dmg", "healing", "obj_dmg"]

        hero_ids = data[:, 0]
        team = data[:, -2]
        labels = data[:, -1:]
        # metadata = np.concatenate((hero_ids, team, labels), axis=1)
        train = data[:, 1:-2]

        print("Train shape: ", train.shape)
        print("Labels shape: ", labels.shape)
        # print("Metadata shape: ", metadata.shape)

        df = pd.DataFrame(data=train, columns=df_columns)
        print("df: ", df.head())
        scaler = MinMaxScaler()
        scaler.fit(df)
        t_df = scaler.transform(df)
        data = t_df

        model = models.Sequential()
        model.add(layers.Dense(8, input_shape=(8,), activation='relu'))
        # model.add(layers.Dense(64, activation='relu'))
        # model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(6, activation='relu'))
        model.add(layers.Dropout(0.05))
        model.add(layers.Dense(4, activation='relu'))
        model.add(layers.Dense(2, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        print(model.summary())
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        training_history = model.fit(
            train, 
            labels, 
            epochs=20, 
            batch_size=50,
            validation_split=0.3
        )
        model.save('deadlock_player_model')
        print("Model saved as deadlock_player_model")
        print("Training history: ", training_history)
        return training_history

    def plot_history(self, history):
        print("Plotting history")
        print(history)
        accuracy = history['accuracy']
        loss = history['loss']
        val_accuracy = history['val_accuracy']
        val_loss = history['val_loss']
        (figure, axes) = plt.subplots(1,2,figsize=(10,4), dpi=300)
        axes[0].plot(accuracy, label="Training")
        axes[0].plot(val_accuracy, label="Validation")
        axes[0].legend()
        axes[0].set_xlabel("Epoch")
        axes[0].grid()
        axes[0].set_title("Accuracy")
        axes[1].plot(loss, label="Training")
        axes[1].plot(val_loss, label="Validation")
        axes[1].legend()
        axes[1].set_xlabel("Epoch")
        axes[1].grid()
        axes[1].set_title("Loss")
        plt.show()


    def train(self, data):

        np.random.shuffle(data)

        amber_teams = data[:, 0]
        saphire_teams = data[:, 1]
        all_teams = np.concatenate((amber_teams, saphire_teams), axis=0)
        print("one team: ", all_teams[0])
        all_players = None
        for team in all_teams:
            for player in team:
                if all_players is None:
                    all_players = player
                else:
                    all_players = np.concatenate((all_players, player), axis=0)

        print("All players: ", all_players.shape)
        print("All players: ", all_players[0])

        

        return 
        # df = pd.DataFrame(data=data,    # values
        #          columns=column_names)  # 1st row as the column names

        # scaler = MinMaxScaler()
        # scaler.fit(df)
        # t_df = scaler.transform(df)
        # data = t_df

        # train = data[:, :-1]
        # labels = data[:, -1:]
        # print("Training Data Shape: ", train.shape)
        # print("Labels Shape: ", labels.shape)

        # model = models.Sequential()
        # model.add(layers.Dense(138, input_shape=(138,), activation='relu'))
        # model.add(layers.Dense(64, activation='relu'))
        # model.add(layers.Dropout(0.2))
        # model.add(layers.Dense(32, activation='relu'))
        # model.add(layers.Dense(8, activation='relu'))
        # model.add(layers.Dense(1, activation='sigmoid'))
        # print(model.summary())
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # training_history = model.fit(
        #     train, 
        #     labels, 
        #     epochs=100, 
        #     batch_size=50,
        #     validation_split=0.2
        # )
        # model.save('ragelock_model')
        # print("Model saved as ragelock_model")
        # return training_history



if __name__ == "__main__":
    main()
