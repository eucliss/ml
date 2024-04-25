import numpy as np
from random import random

class Container:

    def __init__(self, weight, location, size=40, contents="Cars"):
        self.weight = weight
        self.size = size
        self.location = location
        self.contents = contents

class ContainerStack:

    def __init__(self, max_height):
        self.max_height = max_height
        self.stack = []

    def add_container(self, container):
        if len(self.stack) < self.max_height:
            self.stack.append(container)
            return True
        else:
            return False
    
    def remove_container(self):
        if len(self.stack) > 0:
            container = self.stack.pop()
            return container
        else:
            return False
    
    def get_weight(self):
        weight = 0
        for container in self.stack:
            weight += container.weight
        return weight

    def get_height(self):
        return len(self.stack)

    def print_stack(self):
        for container in self.stack:
            print(f'Container: {container.location} Weight: {container.weight}')

class Ship:
    def __init__(self, rows, columns, max_container_height):
        self.rows = rows
        self.columns = columns
        self.max_height = max_container_height
        # self.ship = np.zeros((rows, columns))
        self.ship = np.zeros((rows, columns), dtype=ContainerStack)
        for i in range(0, rows):
            for j in range(0, columns):
                self.ship[i, j] = ContainerStack(max_container_height)
        # self.container_heights = np.zeros((rows, columns))

    def gen_random_stack(self, row, col, count):
        for i in range(0, count):
            self.ship[row, col].add_container(Container(weight=int(random() * 100), location="USA"))

    def get_height(self, row, column):
        return self.ship[row, column].get_height()

    def get_weight(self, row, column):
        return self.ship[row, column].get_weight()

    def add_container(self, row, column, container:Container):
        if self.get_height(row, column) < self.max_height:
            self.ship[row, column].add_container(container)
            return True
        else:
            return False
    
    def remove_container(self, row, column):
        if self.get_height(row, column) > 0:
            container = self.ship[row, column].remove_container()
            return container
        else:
            return False

    def visualize_ship(self):
        print("The shape of the ship is: ", self.ship.shape)
        print("The max height of the containers is ", self.max_height)
        print("-------Bow-------")
        for i in range(0, self.rows):
            for j in range(0, self.columns):
                print(" |", self.get_height(i, j), end="| ")
            print()
        print("-------Stern-----")
        return
    
    def analyze_balance(self):
        result = {}
        bow_stern = self.bow_stern_balance()
        bow_stern_favor = bow_stern['offbalance_favor']
        result['bow_stern'] = bow_stern

        if 0.49 <= bow_stern['bow_ratio'] <= 0.51:
            print("The ship is balanced")
            print("Bow ratio: ", bow_stern['bow_ratio'])
            result['bow_stern_balance'] = "Balanced"
        elif bow_stern_favor == "Bow":
            print("The bow is off balance by", bow_stern['bow'] - bow_stern['stern'])
            print("Bow ratio: ", bow_stern['bow_ratio'])
            result['bow_stern_balance'] = "Bow"

        else:
            print("The stern is off balance by", bow_stern['stern'] - bow_stern['bow'])
            print("Stern ratio: ", bow_stern['stern_ratio'])
            result['bow_stern_balance'] = "Stern"
        
        port_starboard = self.port_starboard_balance()
        port_starboard_favor = port_starboard['offbalance_favor']
        result['port_starboard'] = port_starboard

        if 0.49 <= port_starboard['port_ratio'] <= 0.51:
            print("The ship is balanced")
            print("Port ratio: ", port_starboard['port_ratio'])
            result['port_starboard_balance'] = "Balanced"
        elif port_starboard_favor == "Port":
            print("The port is off balance by", port_starboard['port'] - port_starboard['starboard'])
            print("Port ratio: ", port_starboard['port_ratio'])
            result['port_starboard_balance'] = "Port"
        else:
            print("The starboard is off balance by", port_starboard['starboard'] - port_starboard['port'])
            print("Starboard ratio: ", port_starboard['starboard_ratio'])
            result['port_starboard_balance'] = "Starboard"
        return result



    def port_starboard_balance(self):
        total = 0
        port = 0
        starboard = 0
        for i in range(0, self.rows):
            for j in range(0, self.columns):
                if j < self.columns / 2:
                    port += self.get_weight(i, j)
                    total += self.get_weight(i, j)
                else:
                    starboard += self.get_weight(i, j)
                    total += self.get_weight(i, j)
        port_ratio = port/total
        if port_ratio >= 0.55:
            offbalance_favor = "Port"
        else:
            offbalance_favor = "Starboard"

        result = {
            "total": total,
            "port": port,
            "starboard": starboard,
            "port_ratio": port/total,
            "starboard_ratio": starboard/total,
            "offbalance_favor": offbalance_favor
        }
        return result

    def bow_stern_balance(self):
        total = 0
        bow = 0
        stern = 0
        print(self.rows / 2)
        for i in range(0, self.rows):
            for j in range(0, self.columns):
                if i < self.rows / 2:
                    bow += self.get_weight(i, j)
                    total += self.get_weight(i, j)
                else:
                    stern += self.get_weight(i, j)
                    total += self.get_weight(i, j)
        bow_ratio = bow/total
        if bow_ratio >= 0.55:
            offbalance_favor = "Bow"
        else:
            offbalance_favor = "Stern"

        result = {
            "total": total,
            "bow": bow,
            "stern": stern,
            "bow_ratio": bow/total,
            "stern_ratio": stern/total,
            "offbalance_favor": offbalance_favor
        }
        return result



# s = Ship(5, 3, 3)
# s.gen_random_stack(0, 0, 3)
# s.gen_random_stack(2, 2, 3)
# print(s.ship.shape)
# s.ship[2, 2].print_stack()
# s.visualize_ship()
# for i in range(0, 10):
#     s.add_container(int(random()* 10 % 3), int(random() * 10 %3))
# s.visualize_ship()
    
# stack = ContainerStack(5)
# stack.add_container(Container(weight=100, location="USA"))
# stack.add_container(Container(weight=100, location="CHINA"))
# stack.add_container(Container(weight=100, location="TAIWAN"))
# stack.add_container(Container(weight=100, location="EURPOE"))

# print(stack.get_weight())
# stack.remove_container()
# stack.print_stack()


        