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
import asyncio
import csv

graph_array = []

def benchmark_accuracy(correct, result):
    correct_rows, correct_cols = correct.shape
    result_rows, result_cols = result.shape

    row_diff = abs(correct_rows - result_rows)
    col_diff = abs(correct_cols - result_cols)

    common_rows = min(correct_rows, result_rows)
    common_cols = min(correct_cols, result_cols)

    correct_common = correct[:common_rows, :common_cols]
    result_common = result[:common_rows, :common_cols]

    if common_rows > 0 and common_cols > 0:
        mse_common = np.mean((correct_common - result_common) ** 2)
    else:
        mse_common = 0
                
    extra_rows = abs(correct_rows - result_rows)
    extra_cols = abs(correct_cols - result_cols)
    wrong_cells = (extra_cols * result_rows) + (extra_rows * result_cols)
    total_cells = correct_cols * correct_rows
    penalty_factor = 5

    if total_cells > 0:
        penalty = wrong_cells*penalty_factor
    else:
        penalty = 0

    print(f"Penalty is {penalty}")
                
    final_mse = mse_common + penalty

    comparison_result = {
        'common_mse': float(mse_common),
        'penalty': float(penalty),
        'final_mse': float(final_mse),
        'dimensions': {
            'correct': (correct_rows, correct_cols),
            'result': (result_rows, result_cols),
            'common': (common_rows, common_cols)
        },
        'differences': {
            'missing_rows': max(0, correct_rows - result_rows),
            'extra_rows': max(0, result_rows - correct_rows),
            'missing_cols': max(0, correct_cols - result_cols),
            'extra_cols': max(0, result_cols - correct_cols)
        },
    }
                
    print(comparison_result)
    return final_mse

def benchmark_accuracy_time(results, folders): 
    numbers = list(results.keys())
    mses = {}
    for number in numbers:
        if "test_data/timetest/"+str(number)+"node" in folders:
            correct = []
            try:
                result = results[number]
                correct = np.loadtxt("test_data/timetest/"+str(number)+"node/correct.csv", delimiter=',')

                mses[number] = benchmark_accuracy(correct, result)
 
            except: 
                print("No correct matrix to check against")
    return mses
                


def run(filepath):
    print("Running...")
    global graph_array
    graph_array = asyncio.run(main.agents(filepath)) 
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

folders = get_folders_os_listdir("test_data/timetest")

def benchmark_time():
    folders = get_folders_os_listdir("test_data/timetest")
    folder_times = {}
    results = {}
    for folder in folders:
        number = int((folder.split("/")[2]+"_cleaned.txt").split("edge")[0])
        folder_times[number] = round(timeit.timeit(lambda: run(folder+"/" + folder.split("/")[2] + "_cleaned.txt"), number=1), 4)
        results[number] = main.graphing(graph_array, False)
    
    return (folder_times, results)


benchmark_times, results = benchmark_time()
print(benchmark_times)
print(benchmark_accuracy_time(results, folders))

#benchmark accuracy

grammarfiles = get_files_in_directory("test_data/accuracytest/grammar")
structuralfiles = get_files_in_directory("test_data/accuracytest/structural")
grammarresults = {}
structuralresults = {}
correct = np.loadtxt("test_data/accuracytest/correct.csv", delimiter=',')

for file in grammarfiles:
    print(file)
    graph_array = asyncio.run(main.agents(file))
    result = main.graphing(graph_array, False)
    grammarresults[int(file.split("golden")[1][:-4])] = benchmark_accuracy(correct, result)


for file in structuralfiles:
    print(file)
    graph_array = asyncio.run(main.agents(file))
    result = main.graphing(graph_array, False)
    structuralresults[int(file.split("golden")[1][:-4])] = benchmark_accuracy(correct, result)


structuralresults = dict(sorted(structuralresults.items()))
grammarresults = dict(sorted(grammarresults.items()))
print(grammarresults)
print(structuralresults)

x1 = list(grammarresults.keys())
y1 = list(grammarresults.values())

x2 = list(structuralresults.keys())
y2 = list(structuralresults.values())

plt.plot(x1, y1, marker='o', label="Spelling errors")
plt.plot(x2, y2, marker='o', label="Structural errors")

plt.xlabel("No. of errors")
plt.ylabel("Mean square error with penalty")
plt.title("Accuracy benchmark")
plt.legend()
plt.show()

benchmark_times = dict(sorted(benchmark_times.items()))
x = list(benchmark_times.keys())
y = list(benchmark_times.values())
plt.plot(x, y, marker='o')
plt.xlabel("No. of edges")
plt.ylabel("Time taken/s")
plt.title("Time taken benchmark")
plt.show()
