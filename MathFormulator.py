import re
from metagpt.actions import Action
from metagpt.roles import Role
import asyncio
from metagpt.context import Context
from metagpt.logs import logger
from metagpt.schema import Message

'''
The objective function must assign variables to nodes but not edges, 
it should partition the nodes by assigning a binary value 1 or 0 to each node to group them into sets,
and when an edge's endpoints are of 2 different sets, you can tell that it is cut.
'''

class MathFormulation(Action):
    PROMPT_TEMPLATE: str = """
    Based on the following nodes, edges and weights provided, 
    {instruction}, 
    create a mathematical formulation of this max-cut problem in the form of an objective function,
    such that the nodes are partitioned into 2 groups such that the sum of weights of the edges is maximised.
    Present it using this format w_ij*(x_i + x_j - 2*x_i*x_j),
    where x_i represents one node, x_j represents the other, and w_ij represents the weight of the edge ij
    Return ```objective_function your_objective_function_here ``` with NO other texts,
    your function:
    """

    name: str = "MathFormulation"

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

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

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  # todo will be SimpleWriteCode()

        msg = self.get_memories(k=1)[0]  # find the most recent messages
        code_text = await todo.run(msg.content)
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg
    
async def main():
    msg = '''
    10 nodes v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, 
    with weights of edges as follows,
    edge v1v2 has weight 5,
    edge v1v3 has weight 3,
    edge v1v5 has weight 4,
    edge v2v4 has weight 6,
    edge v2v6 has weight 2,
    edge v3v4 has weight 7,
    edge v3v7 has weight 5,
    edge v4v8 has weight 8,
    edge v5v6 has weight 3,
    edge v5v9 has weight 4,
    edge v6v10 has weight 6,
    edge v7v8 has weight 2,
    edge v8v9 has weight 7,
    edge v9v10 has weight 5,
    edge v1v10 has weight 1
    '''
    context = Context()
    role = MathFormulator(context=context)
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)

asyncio.run(main())