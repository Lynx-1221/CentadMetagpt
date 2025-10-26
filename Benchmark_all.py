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
import os

graph_array = []

files = [
    "test1.txt",
    "test2.txt"
]

def run(filepath):
    print("Running...")
    global graph_array
    graph_array = fire.Fire(main.agents, command=[filepath]) 
    print("Run Complete")

def get_folders_os_listdir(directory_path):
    folders = []
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isdir(full_path):
            folders.append(full_path)
    return folders

def get_files_in_directory(directory):
    file_names = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path):
            file_names.append(full_path)
    return file_names

folders = get_folders_os_listdir("test_data")
correct = ""
folder_times = {}

for folder in folders:
    execution_times = {}
    files = get_files_in_directory(folder)
    correct = None
    try:
        execution_times[folder.split("/")[1]+"_cleaned.txt"] = round(timeit.timeit(lambda: run(folder+"/" + folder.split("/")[1] + "_cleaned.txt"), number=1), 4)
        files.remove(folder+"/" + folder.split("/")[1] + "_cleaned.txt")
        correct = graph_array.copy()
    except:
        print("No clean file")
    for i in files:
        try:
            print("Executing")
            execution_time = timeit.timeit(lambda: run(i), number=1)
            temp_array = graph_array.copy()
            values = [str(round(execution_time, 4))+"s"]
            if correct != None:
                values.append(temp_array == correct)
            execution_times[i.replace(folder+"/", "")] = values
        except:
            print("Failed to process")
    folder_times[folder] = execution_times

print(folder_times)


'''
# main.graphing(graph_array)
'''