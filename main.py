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
app = typer.Typer()

graph_array = []

def extract_array(rsp):
    result = []
    rsp = rsp.split(" ")
    for i in rsp:
        temp = i[1:-1].split(",")
        temp[2] = int(temp[2])
        temp = tuple(temp)
        result.append(temp)
    return result

#Start with an agent to turn the real world situation to a combinatorial problem then do math formulation


class ProblemFormulation(Action):
    PROMPT_TEMPLATE: str = """
    Given the following word problem, 
    {instruction},
    create a continuous string of edges seperated by spaces in the format (x1,y1,w1) (x2,y2,w2),
    where the x and y represent the nodes that the lines connect, and w represents the weight of those edges,
    Return ```Problem: your_nodes_here ```with NO other texts,
    Your edges:
    """

    name: str = "ProblemFormulation"

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        answer = ProblemFormulation.parse_code(rsp)

        return answer

    @staticmethod
    def parse_code(rsp):
        pattern = r"```Problem:([\s\S]*?)```"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp

        global graph_array
        graph_array = extract_array(code_text.split("Problem: ")[1])

        return code_text
    
class ProblemFormulator(Role):
    name: str = "David"
    profile: str = "ProblemFormulator"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch({UserRequirement})
        self.set_actions([ProblemFormulation])

    async def _act(self) -> Message:
        ans = await self.rc.todo.run(self.rc.history)
        if ans[:7] == "Problem":
            self.rc.env.publish_message(Message(content=ans, cause_by=ProblemFormulation, send_to=MathFormulator)) 
        return None


class MathFormulation(Action):

    #can use latex format from this step forwards

    PROMPT_TEMPLATE: str = """
    Based on the following edge list provided, 
    {context}, which is in the format (x,y,w) where x and y are the two nodes that are connected by the edge,
    and w is the weight of the edge,
    create a mathematical formulation of this max-cut problem in the form of an objective function,
    such that the nodes are partitioned into 2 groups such that the sum of weights of the edges is maximised.
    Present it using this format w_ij*(x_i + x_j - 2*x_i*x_j),
    where x_i represents one node, x_j represents the other, and w_ij represents the weight of the edge ij
    Return ```objective_function your_objective_function_here ``` with NO other texts,
    your function:
    """

    name: str = "MathFormulation"

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context)

        rsp = await self._aask(prompt)

        create_function = MathFormulation.parse_code(rsp)

        return create_function

    @staticmethod
    def parse_code(rsp):
        pattern = r"```objective_function(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text
    
class MathFormulator(Role):
    name: str = "Alice"
    profile: str = "MathFormulator"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([MathFormulation]) 
        self._watch({ProblemFormulation})

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  

        context = self.rc.history
        msg = await todo.run(context)
        self.rc.env.publish_message(Message(content=msg, cause_by=MathFormulation, send_to=QUBOTermFormulator)) 

# can let agents use external tools to convert from objective function to qubo terms
# should include tests at each step to make sure result is right

class FormingQUBOTerms(Action):
    PROMPT_TEMPLATE: str = """
    Given the following objective function,
    {context}, 
    break it down into linear and quadratic terms,
    returning the various values with their Qx,y values,
    Return ```QUBO_values your_values_here ``` at the end,
    your values:
    """

    name: str = "FormingQUBOTerms"

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context)

        rsp = await self._aask(prompt)

        create_function = FormingQUBOTerms.parse_code(rsp)

        return create_function

    @staticmethod
    def parse_code(rsp):
        pattern = r"```QUBO_values(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text
    
class QUBOTermFormulator(Role):
    name: str = "Yamada"
    profile: str = "QUBOTermFormulator"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([FormingQUBOTerms])
        self._watch({MathFormulation})
        

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  

        context = self.rc.history

        msg = await todo.run(context)
        self.rc.env.publish_message(Message(content=msg, cause_by=FormingQUBOTerms, send_to=QuboMatrixFormulator)) 

# can change to numpy array formulator or use similar libraries

class QuboMatrixFormulation(Action):
    PROMPT_TEMPLATE: str = """
    Given the following qubo values,
    {context}, 
    make a qubo matrix with it,
    summing together every value,
    Return ```QUBO_matrix your_matrix_here ``` at the end,
    your matrix:
    """

    name: str = "QuboMatrixFormulation"

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context)

        rsp = await self._aask(prompt)

        create_function = QuboMatrixFormulation.parse_code(rsp)

        return create_function

    @staticmethod
    def parse_code(rsp):
        pattern = r"```QUBO_matrix(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text
    
class QuboMatrixFormulator(Role):
    name: str = "Klein"
    profile: str = "QuboMatrixFormulator"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([QuboMatrixFormulation])
        self._watch({FormingQUBOTerms})
        

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  

        context = self.get_memories()

        code_text = await todo.run(context)
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg


async def main():
    #replace idea with the code to put the file in.
    problem = '''
    Scenario: A company has 8 office clusters (nodes) that run different parts of a distributed application. Pairs of clusters communicate with known average bandwidth demand (edge weights, in Mbps). Each cluster also has a processing-capacity rating (node value, in arbitrary processing units). The IT team wants to split the 8 clusters into two groups (Group X and Group Y) so that cross-group communication is maximized — i.e.,
    the sum of bandwidth of links that go between the groups is as large as possible (a weighted max-cut). 
    Because latency-sensitive tasks should remain balanced across the two sites,
    the total processing capacity in the two groups should not differ by more than 10 units.
Nodes (cluster name — processing capacity):
Alpha — 18
Beta — 12
Gamma — 15
Delta — 10
Epsilon — 14
Zeta — 9
Eta — 13
Theta — 11
Edges (unordered pair — bandwidth demand in Mbps):
(Alpha, Beta) — 30
(Alpha, Gamma) — 20
(Alpha, Delta) — 5
(Alpha, Epsilon) — 12
(Beta, Gamma) — 25
(Beta, Delta) — 8
(Beta, Zeta) — 10
(Gamma, Epsilon) — 18
(Gamma, Eta) — 7
(Delta, Zeta) — 22
(Epsilon, Zeta) — 6
(Epsilon, Theta) — 14
(Zeta, Eta) — 11
(Eta, Theta) — 9
(Delta, Theta) — 4
(Any pair not listed has bandwidth demand 0.)
    '''
    idea: str = problem
    context = Context() # Load config2.yaml
    env = Environment(context=context)
    env.add_roles([ProblemFormulator(), MathFormulator()])
    env.publish_message(Message(content=idea, send_to=ProblemFormulator)) # Send the user's message to Agent A to start the process.
    while not env.is_idle: # `env.is_idle` becomes True only when all agents have no new messages to process.
        await env.run()


if __name__ == "__main__":
    fire.Fire(main)
    G = nx.Graph()
    G.add_weighted_edges_from(graph_array)
    nx.draw_planar(G, with_labels=True)
    plt.show()