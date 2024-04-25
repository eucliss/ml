from containers import Container, ContainerStack, Ship
from random import random

rows = 20
columns = 10
max_container_height = 9

max_teus = rows * columns * max_container_height
print(max_teus)

ship = Ship(rows, columns, max_container_height)
ship.visualize_ship()

ports = ["NYC", "CHS", "TPA", "LDN", "UAE", "HKG", "SIN", "SYD"]

# Generate all the shipping containers
for i in range(0, int(max_teus/2)):
    container = Container(weight=int(random() * 100), location=ports[int(random() * len(ports))])
    added = False
    while not added:
        row = int(random() * rows)
        column = int(random() * columns)
        added = ship.add_container(row, column, container)
        
ship.visualize_ship()
result = ship.bow_stern_balance()
result = ship.port_starboard_balance()
result = ship.analyze_balance()
print(result)



# new_ship = Ship(6, 4, 1)
# for i in range(0 ,4):
#     container = Container(weight=100, location="USA")
#     new_ship.add_container(0, i, container)
#     new_ship.add_container(1, i, container)
#     new_ship.add_container(2, i, container)
# new_ship.visualize_ship()
# result = new_ship.bow_stern_balance()
# print(result)
