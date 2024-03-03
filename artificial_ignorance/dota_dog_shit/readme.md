# Dota Dogshit

## The ignorant AI model that tells you if your team comp is dogshit or not

## Disclaimer:

This is for learning purposes only.

This model is not optimized or trained on data that is cleaned up and good. Theres a reason its called an ignorant model - it doesnt really know what its doing and doesnt know why the team comp would be good or not. Additionally, it has no game context so it has no idea what team comp it was playing against. Maybe V2 will retrain on radiant v dire and the outcome.

### Model details
This dogshit model is trained on as many pro games as I could scrape from the OpenDota API. I think there are around 2500 matches that I pulled down using dota.py.

Matches are loaded into a mongo DB along with the heroes objects that OpenDota has available.

Matches are then split into Radiant and Dire teams, meaning each match gives us 2 data points:
- Radiant, win/lose
- Dire, win/lose
Each team is then put through data augmentation where it just randomizes the order of the heroes and creates more data points.

The teams are multi-hot encoded into a 138 length vector, 1 for each hero. (actually there arent 138 heroes, but the OpenDota API uses ID numbers and I did a simple map from ID to column, some columns are completely empty - should probably fix this too).

The model is a simple feed forward neural network with 2 hidden layers. The input layer is 138 nodes, 1 for each input. The output layer is 1 node, higher than .50 points us towards a win, lower than .50 points us towards a loss.

### How to use

Dota.py is used for scraping down data.
DotaDatabase.py is used for loading data into a mongo DB.
dota_dog_shit is used for building and training the model
(Nice file names fr, 0 standardization)

Requirements:
- Load data into mongodb from dota.py

I think thats it, pretty much need to pyll down the data and then load it into the MongoDB, from there DotaDogShit class should deal with pulling the stuff in and normalizing it all and parsing it.

You can predict a team by doing:

```
dogshit = DotaDogShit()
team = ['Lina', 'Meepo', 'Hoodwink', 'Mars', 'Pugna']
res = dogshit.predictTeam(team)

# res = EZ Midler
# Probably true in this meta, this team is not bad
```

GL HF