from DotaDatabase import DotaDatabase
from dota import Dota
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import models, layers, datasets, optimizers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DotaDogShit():
    # A class to see if your team is utter dogshit


    def __init__(self):
        print("Loading up the dota dogshit model ...")
        self.dota = Dota()
        self.database = DotaDatabase()
        self.matches = self.database.getAllRecords('matches')
        self.heroes = self.database.getAllRecords('heroes')
        try:
            self.training_data = self.loadTrainingData()
        except:
            self.training_data = None
            print("No training data found - building a new set.")
            self.training_data = self.buildTrainingSet()
        try:
            self.model = models.load_model('dota_model')
        except:
            self.model = None
            print("No model found, training a new one")
            self.training_history = self.train(self.training_data)
            self.model = models.load_model('dota_model')

    def saveTrainingData(self, data):
        with open('../data/dota_matches.npy', 'wb') as f:
            np.save(f, data, allow_pickle=True)
        return
    
    def loadTrainingData(self):
        with open('../data/dota_matches.npy', 'rb') as f:
            return np.load(f, allow_pickle=False)

    def buildTrainingSet(self, data_augemntation=True, aug_count=3):
        # Builds a training set from the matches in the database
        print("Building Training Set, count: ", len(self.matches))
        res = np.array([])
        for match in self.matches:
            print("Processing match: ", match['match_id'])
            match_team_data = self.buildTrainingObjectsForMatch(match)
            if match_team_data.size == 0:
                continue
            if data_augemntation:
                augmented_match = np.array([])
                for team in match_team_data:
                    augmented_team = self.data_augementation(team, count=aug_count)
                    if augmented_match.size == 0:
                        augmented_match = augmented_team
                    else:
                        augmented_match = np.append(augmented_match, augmented_team, axis=0)
                match_team_data = augmented_match

            if res.size == 0:
                res = np.array(match_team_data)
            else:
                res = np.append(res, match_team_data, axis=0)
        
        res = self.vectorize(res, res.shape[1] - 1)
        self.saveTrainingData(res)
        return res
    
    def vectorize(self, data, dimension):
        # print("Vectorizing Data")
        # print("Data Shape: ", data.shape)
        # print("Dimension: ", dimension)
        # IDs for heroes go up to 138
        multihot = np.zeros((len(data), 139))
        # print("Multihot Shape: ", multihot.shape)
        for i, team in enumerate(data):
            # Multi hot the heroes
            for hero in team[:-1]:
                multihot[i, int(hero)] = 1.0
            # Track the outcome from the input
            multihot[i, -1] = team[-1]
        return multihot
   
    def UniqueData(self, data):
        return np.unique(data, axis=0)

    def getTeams(self, match):
        radiant_id = 0
        try:
            picks_bans = match['picks_bans']
            if picks_bans == None:
                return [], []
        except:
            return [], []
        radiant = []
        dire = []
        for pick in picks_bans:
            if pick['is_pick']:
                if pick['team'] == radiant_id:
                    radiant.append(pick['hero_id'])
                else:
                    dire.append(pick['hero_id'])
        return radiant, dire
    
    def oneHotWin(self, match):
        radiant_win = match['radiant_win']
        if radiant_win == 1:
            return 1.0, 0.0
        else:
            return 0.0, 1.0

    def data_augementation(self, team_object, count=3):
        # Takes a team and its label and shuffles the team to augment data.
        # Returns original and shuffled in an np array
        res = np.array([team_object])
        team, outcome = team_object[:-1], team_object[-1]
        for i in range(count):
            train_obj = np.array(np.random.permutation(team))
            train_obj = np.concatenate((train_obj, [outcome]), axis=0)
            res = np.append(res, [train_obj], axis=0)
        return res

    def buildTrainingObjectsForMatch(self, match):
        # Takes a match and returns a numpy array with radiant and dire objects
        radiant, dire = self.getTeams(match)
        if len(radiant) == 0 or len(dire) == 0:
            return np.array([])
        radiant_win, dire_win = self.oneHotWin(match)
        radiant = np.array(radiant + [radiant_win])
        dire = np.array(dire + [dire_win])
        return np.array([radiant, dire])
    
    def split_train_and_labels(self, data):
        return data[:, :-1], data[:, -1:]
    
    def decodeWinLose(self, outcome):
        outcome = outcome[-1]
        return "Win" if outcome == 1.0 else "Lose"

    def decodeTeam(self, match):
        team = match[:-1]
        hero_dict = {}
        for hero in team:
            record = self.database.getRecord({'id': int(hero)}, collection='heroes')
            hero_dict[hero] = record[0]['localized_name']
        return [hero_dict[hero] for hero in team]

    def encodeTeam(self, team):
        hero_dict = {}
        for hero in team:
            record = self.database.getRecord({'localized_name': hero}, collection='heroes')
            hero_dict[hero] = record[0]['id']
        return [hero_dict[hero] for hero in team]
    
    def train(self, data):
        np.random.shuffle(data)

        train, test = self.split_train_and_labels(data)
        
        print(train)
        print(test)
        column_names = [f"{i}" for i in range(0, 139)]
        column_names[-1] = "Outcome"
        df = pd.DataFrame(data=data,    # values
                 columns=column_names)  # 1st row as the column names

        scaler = MinMaxScaler()
        scaler.fit(df)
        t_df = scaler.transform(df)
        data = t_df

        train = data[:, :-1]
        labels = data[:, -1:]
        print("Training Data Shape: ", train.shape)
        print("Labels Shape: ", labels.shape)

        model = models.Sequential()
        model.add(layers.Dense(138, input_shape=(138,), activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(8, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        print(model.summary())
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        training_history = model.fit(
            train, 
            labels, 
            epochs=100, 
            batch_size=50,
            validation_split=0.2
        )
        model.save('dota_model')
        print("Model saved as dota_model")
        return training_history
    
    def plot_history(self, history):
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

    def predict(self, model, data):
        return model.predict(data)
    
    def decodePrediction(self, prediction):
        prediction = prediction[0]
        if prediction < 0.001:
            return "Jark Q"
        if prediction < 0.2:
            return "Dog Shit"
        if 0.2 < prediction < 0.4:
            return "Horse Water"
        if 0.4 < prediction < 0.6:
            return "buh"
        if 0.6 < prediction < 0.8:
            return "wug"
        else:
            return "EZ Midler"

    def testModel(self):
        heroes = [i['localized_name'] for i in self.heroes]
        teams = []
        for i in range(100):
            random_team = (np.random.choice(heroes, 5))
            np.unique(random_team)
            while len(random_team) != 5:
                random_team = (np.random.choice(heroes, 5))
                random_team = np.unique(random_team)
            teams.append(np.random.choice(heroes, 5))

        predictions = {}
        model = models.load_model('dota_model')
        for team in teams:
            print("Team: ", team)
            team_string = str(team)
            team = self.encodeTeam(team)
            team = self.vectorize(np.array([team]), 5)
            team = team[:, :-1]
            # vectorized_teams.append(team)
        
            prediction = model.predict(team)
            predictions[team_string] = prediction
        for match in predictions.keys():
            print("Team: ", match)
            print("Prediction: ", self.decodePrediction(predictions[match]))
            print("-------------------")

    def predictTeam(self, team):
        print("Predicting outcome for team: ", team)
        team = self.encodeTeam(team)
        team = self.vectorize(np.array([team]), 5)
        team = team[:, :-1]
        res = self.decodePrediction(self.predict(self.model, team))
        if res in ['wug', 'EZ Midler']:
            print("This team is not dogshit")
            print("Prediction: ", res, "(Win)")
        elif res in ['buh']:
            print("This team is sometimes maybe good sometimes maybe shit")
            print("Prediction: ", res, "(Good luck, Have fun)")
        else:
            print("This team is dogshit")
            print("Prediction: ", res, "(Trash comp, ez loss)")
        return res

dogshit = DotaDogShit()

team = ['Windranger', 'Templar Assassin', 'Kunkka', 'Lion', 'Phoenix']
res = dogshit.predictTeam(team)

team = ['Undying', 'Arc Warden', 'Slardar', 'Snapfire', 'Timbersaw']
res = dogshit.predictTeam(team)

# dogshit.testModel()