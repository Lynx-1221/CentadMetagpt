import main
import fire
import re
import typer
import networkx as nx
import matplotlib.pyplot as plt
from metagpt.logs import logger
from metagpt.team import Team, Environment
from metagpt.schema import Message
from metagpt.actions import Action, UserRequirement
from metagpt.roles import Role
from metagpt.context import Context
import numpy as np
import timeit

graph_array = []

files = [
    "test1.txt",
    "test2.txt"
]

execution_times = []

def run(filepath):
    print("Running...")
    global graph_array
    graph_array = fire.Fire(main.agents, command=[filepath]) 

for i in files:
    try:
        execution_time = timeit.timeit(lambda: run("test_data/"+i), number=1)
        execution_times.append(i+" : "+str(execution_time)+"s")
    except:
        print("Failed to process")

print(execution_times)
# main.graphing(graph_array)