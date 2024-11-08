import sys
import json
sys.path.append("..")
import jsonschema
from jsonschema import validate

import random
from faker import Faker
from typing import Dict, Any, List


class Generator():
    def __init__(self, match_id: int = 1):
        # self.db = Database("ragelock_1")
        with open('match_schema.json', 'r') as schema_file:
            self.schema = json.load(schema_file)
        self.fake = Faker()
        self.heroes = self.heroes()
        self.match_id = match_id

    def generate_rage_player(self, hero: str, duration: int, index: int):
        leave_timer = random.randint(0, int(duration/2))
        leave_percentage = leave_timer / duration  
        initial_player = self.generate_player(hero, duration, index)

        souls_per_minute = int(initial_player["souls_per_minute"] * 0.3 * leave_percentage)
        total_souls = int(duration / 60 * souls_per_minute)
        kills = self.fake.random_int(min=0, max=8)
        deaths = self.fake.random_int(min=5, max=20)
        assists = self.fake.random_int(min=0, max=8)
        player_dmg = int(initial_player["player_dmg"] / 2 * leave_percentage )
        object_dmg = int(initial_player["object_dmg"] / 2 * leave_percentage)
        healing = int(initial_player["healing"] / 2 * leave_percentage)
        return {
            "steam_name": "rage_" + initial_player["steam_name"],
            "steam_id": initial_player["steam_id"],
            "hero": hero,
            "total_souls": total_souls,
            "souls_per_minute": souls_per_minute,
            "kills": kills,
            "deaths": deaths,
            "assists": assists,
            "player_dmg": player_dmg,
            "object_dmg": object_dmg,
            "healing": healing
        }

    def generate_player(self, hero: str, duration: int, index: int):
        duration_mins = duration / 60
        seed = index / random.randint(1, 12)

        souls_per_minute = self.fake.random_int(min=600, max=1500)
        total_souls = int(duration_mins * souls_per_minute)
        return {
            "steam_name": self.fake.user_name(),
            "steam_id": self.fake.random_int(min=10000000, max=999999999),
            "hero": hero,
            "total_souls": total_souls,
            "souls_per_minute": souls_per_minute,
            "kills": self.fake.random_int(min=0, max=int(25*seed)),
            "deaths": self.fake.random_int(min=0, max=int(20*seed)),
            "assists": self.fake.random_int(min=0, max=int(30*seed)),
            "player_dmg": self.fake.random_int(min=0, max=int(100000*seed)),
            "object_dmg": self.fake.random_int(min=0, max=int(20000*seed)),
            "healing": self.fake.random_int(min=0, max=int(50000*seed))
        }

    def generate_team(self, heroes: List[str], duration: int,rage_index: int = None):
        players = []
        for i, hero in enumerate(heroes):
            if i == rage_index:
                players.append(self.generate_rage_player(hero, duration, i+1))
            else:
                players.append(self.generate_player(hero, duration, i+1))
        
        total_souls = sum([player["total_souls"] for player in players])
        return {
            "players": players,
            "team_souls": total_souls
        }

    def generate_match_heroes(self):
        return self.fake.random_elements(elements=self.heroes, length=12, unique=True)

    def generate_example_match(self) -> Dict[str, Any]:
        # 5m - 1h
        duration = random.randint(300, 3600)

        match_heroes = self.generate_match_heroes()
        amber_hand_heroes = match_heroes[:6]
        saphire_flame_heroes = match_heroes[6:]
        rage_index = random.randint(0, 5)
        rage_team = random.randint(0, 1)
        
        # Rage in this game?
        rage_percentage = 50
        rage_included = random.randint(0, 100) < rage_percentage
        if not rage_included:
            rage_index = -1

        if rage_team == 0:
            amber_team = self.generate_team(amber_hand_heroes, duration, rage_index)
            saphire_team = self.generate_team(saphire_flame_heroes, duration)
        else:
            saphire_team = self.generate_team(saphire_flame_heroes, duration, rage_index)
            amber_team = self.generate_team(amber_hand_heroes, duration)

        self.match_id += 1
        return {
            "match_id": self.match_id - 1,
            "amber_hand": amber_team,
            "saphire_flame": saphire_team,
            "winner": self.fake.random_element(elements=["amber_hand", "saphire_flame"]),
            "duration": duration,
            "rage_status": rage_included
        }
    
    def print_match_data(self, match_data):
        print(f"Match ID: {match_data['match_id']}")
        print(f"Winner: {match_data['winner']}")
        print(f"Rage Status: {match_data['rage_status']}")
        print("Amber Hand:")
        for player in match_data['amber_hand']['players']:
            print(f"{player['steam_name']} {player['hero']} {player['total_souls']} {player['souls_per_minute']} {player['kills']}/{player['deaths']}/{player['assists']} {player['player_dmg']} {player['object_dmg']} {player['healing']}")
        print("Saphire Flame:")
        for player in match_data['saphire_flame']['players']:
            print(f"{player['steam_name']} {player['hero']} {player['total_souls']} {player['souls_per_minute']} {player['kills']}/{player['deaths']}/{player['assists']} {player['player_dmg']} {player['object_dmg']} {player['healing']}")

    def validate_match_data(self, match_data):
        try:
            validate(instance=match_data, schema=self.schema)
            return True
        except jsonschema.exceptions.ValidationError as ve:
            print(f"JSON validation error: {ve}")
            return False
        
    def heroes(self):
        return [
            "abrams",
            "bebop",
            "dynamo",
            "grey talon",
            "haze",
            "infernus",
            "ivy",
            "kelvin",
            "lady geist",
            "lash",
            "mcginnis",
            "mirage",
            "mo & krill",
            "paradox",
            "pocket",
            "seven",
            "shiv",
            "vindicta",
            "viscous",
            "warden",
            "wraith",
            "yamato"
        ]
    
    def get_hero_id(self, hero_name):
        return self.heroes.index(hero_name)

        
# if __name__ == "__main__":
#     g = Generator()
#     res = g.generate_example_match()
#     # g.print_match_data(res)
