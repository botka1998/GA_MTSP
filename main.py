# This is a sample Python script.
import time

import matplotlib.pyplot as plt
import numpy as np

from utils import *
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def init_robots(config: json):
    rob_dict = {}

    for robot in config["robots"]:
        name = robot["name"]
        ip = robot["ip"]
        port = robot["port"]
        toolData = robot["tool_data"]
        wobj = robot["wobj"]

        newRobot = abb.Robot(
            ip=ip,
            port_motion=port,
            port_logger=port + 1
        )

        newRobot.set_tool(toolData)
        newRobot.set_workobject(wobj)

        rob_dict[name] = newRobot
        time.sleep(1)
    return rob_dict

def is_reachable(target):
    is_reachable = True
    x, y, z = target[0], target[1], target[2]
    odgovor = ""
    odgovor = str(robots["ROB1"].isReachable(x, y, z))
    if odgovor == "False":
        is_reachable = False
    odgovor = ""
    odgovor = str(robots["ROB2"].isReachable(x, y, z))
    if odgovor == "False":
        is_reachable = False

    return is_reachable

def check_targets():

    reachable_targets = Globals.targets
    ol = Globals.num_targets
    for i in range(ol):
        if not is_reachable(Globals.targets[i]):
            reachable_targets = np.delete(Globals.targets,i,0)
            Globals.num_targets -= 1
    Globals.targets = reachable_targets
    if ol > Globals.num_targets:
        print("{} tacaka je uklonjeno iz optimizacije".format(ol-Globals.num_targets))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file = open("config.json")
    config_json = json.load(file)
    robots = init_robots(config_json)
    robots["ROB1"].set_speed(speed=[600, 200, 200, 200])
    robots["ROB2"].set_speed(speed=[600, 200, 200, 200])

    check_targets()
    pop = initial_population()
    global_best = get_best(pop)
    global_fitness = get_fitness(global_best)
    fitness = []
    for i in range(Globals.num_generations):
        pop = evolve_population(pop)
        local_best = get_best(pop)
        local_fitness = get_fitness(local_best)
        fitness.append(local_fitness)
        if local_fitness > global_fitness:
            global_best = local_best
            global_fitness = local_fitness

    print("\n\n\nNajbolje resenje:")
    print(global_best)
    print(global_fitness)

    plot(global_best)
    plt.plot(np.arange(Globals.num_generations), fitness, 'r-')
    plt.show()
    robots["ROB1"].set_cartesian([Globals.home1, [0.5, 0, 0.86603, 0]])
    robots["ROB2"].set_cartesian([Globals.home2, [0.5, 0, 0.86603, 0]])
    if global_best[0] != 0:
        base = global_best[2:]
        for i in range(Globals.num_targets):
            robots["ROB1"].set_cartesian([Globals.targets[base[i]-1],[0,0,1,0]])
        robots["ROB1"].set_cartesian([Globals.home1, [0.5,0,0.86603,0]])
    if global_best[1] != 0:
        base = global_best[global_best[0]:]
        for i in range(Globals.num_targets):
            robots["ROB2"].set_cartesian([Globals.targets[base[i]-1],[0,0,1,0]])
        robots["ROB2"].set_cartesian([Globals.home2 , [0.5,0,0.86603,0]])



