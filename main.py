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
from pyqubo import Binary
import asyncio

# app = typer.Typer()

graph_array = []

def clean(text):
    prev = ""
    for i in range(len(text)):
        if text[i] == " " and prev != ")":
            text = text[:i] + "_" + text[i+1:]
        prev = text[i]
    return text

def extract_array(rsp):
    result = []
    rsp = rsp.split(" ")
    for i in rsp:
        temp = i[1:-1].split(",")
        temp[2] = float(temp[2])
        temp = tuple(temp)
        result.append(temp)
    return result



def edges_to_qubo(num_nodes, edges, one_indexed=False):
    Q = np.zeros((num_nodes, num_nodes))
    offset = 1 if one_indexed else 0
    for i, j, w in edges:
        
        u, v = i - offset, j - offset
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            raise ValueError(f"Invalid node index in edge ({i}, {j}) for {num_nodes} nodes.")
        
        Q[i, i] -= w
        Q[j, j] -= w
        Q[i, j] += 2 * w
        Q[j, i] += 2 * w
    
    return Q

class ProblemFormulation(Action):
    PROMPT_TEMPLATE: str = """
    Given the following word problem, 
    {instruction},
    create a continuous string of edges seperated by spaces in the format (x1,y1,w1) (x2,y2,w2),
    where the x and y represent the nodes that the lines connect, and w represents the weight of those edges,
    make sure it is in that format,
    Return 'Problem: your_edges_here',
    """

    name: str = "ProblemFormulation"

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        answer = ProblemFormulation.parse_code(rsp)

        return answer

    @staticmethod
    def parse_code(rsp):
        pattern = r"'Problem:([\s\S]*?)'"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        code_text = clean(code_text)

        return code_text
    
class ProblemFormulator(Role):
    name: str = "David"
    profile: str = "ProblemFormulator"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch({UserRequirement})
        self.set_actions([ProblemFormulation])

    async def _act(self) -> Message:
        todo = self.rc.todo 

        msg = self.get_memories(k=1)[0] 
        code_text = await todo.run(msg.content)
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg


async def agents(filepath="test_data/timetest/10node/10node_cleaned.txt"):
    #replace idea with the code to put the file in.
    f = open(filepath)
    problem = f.read()
    f.close()
    idea: str = problem
    context = Context()
    role = ProblemFormulator(context=context)
    # logger.info(idea)
    result = str(await role.run(idea))
    # logger.info(result)
    graph_array = extract_array(result.split("Problem:_")[1])
    return graph_array


#Function to turn the graph array into a network x graph
def graphing(graph_array, graph=True):
    #Generates graph
    G = nx.Graph()
    # Adds edges with text labels for nodes
    G.add_weighted_edges_from(graph_array)
    #Gets number of nodes
    num_nodes = G.number_of_nodes()

    #Puts nodes into a list
    node_list = list(G.nodes)
    #Turn nodes into integers for numpy matrix formulation
    edge_list = []
    for i in graph_array:
        edge_list.append((node_list.index(i[0]), node_list.index(i[1]), i[2]))
    arr = edges_to_qubo(num_nodes, edge_list)

    # np.savetxt("correct.csv", arr, delimiter=",")
    
    # model = edgelist_string_nodes_to_qubo(graph_array)
    # Draws the graph
    if graph:
        nx.draw_spring(G, with_labels=True)
        plt.show()

    return arr

if __name__ == "__main__":
    asyncio.run(agents("test_data/accuracytest/golden0.txt"))
    graphing(graph_array) 

    
    