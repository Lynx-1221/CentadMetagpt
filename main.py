import fire
import re
import typer
from metagpt.logs import logger
from metagpt.team import Team, Environment
from metagpt.schema import Message
from metagpt.actions import Action, UserRequirement
from metagpt.roles import Role
from metagpt.context import Context
app = typer.Typer()


class ProblemCheck(Action):
    PROMPT_TEMPLATE: str = """
    Given the following max-cut problem:
    {instruction}.
    Return "Problem: {instruction}" if the provided problem is a max-cut problem with enough information to create a mathematical formulation.
    Otherwise, return the missing information or the right format to present the problem. 
    Do not include any additional texts.
    Your answer:
    """

    name: str = "ProblemCheck"

    async def run(self, instruction: str):
        prompt = self.PROMPT_TEMPLATE.format(instruction=instruction)

        rsp = await self._aask(prompt)

        answer = ProblemCheck.parse_code(rsp)

        return answer

    @staticmethod
    def parse_code(rsp):
        pattern = r"```objective_function(.*)```"
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text
    
class ProblemChecker(Role):
    name: str = "David"
    profile: str = "ProblemChecker"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch({UserRequirement})
        self.set_actions([ProblemCheck])

    async def _act(self) -> Message:
        ans = await self.rc.todo.run(self.rc.history)
        if ans[:7] == "Problem":
            self.rc.env.publish_message(Message(content=ans, cause_by=ProblemCheck)) 
        return None


class MathFormulation(Action):
    PROMPT_TEMPLATE: str = """
    Based on the following nodes, edges and weights provided, 
    {context}, if it is not a proper max cut problem then return no. Otherwise
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
        self._watch({ProblemCheck})

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  

        context = self.get_memories()

        code_text = await todo.run(context)
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg


async def main():
    idea: str = '''
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
    context = Context() # Load config2.yaml
    env = Environment(context=context)
    env.add_roles([ProblemChecker(), MathFormulator()])
    env.publish_message(Message(content=idea, send_to=ProblemChecker)) # Send the user's message to Agent A to start the process.
    while not env.is_idle: # `env.is_idle` becomes True only when all agents have no new messages to process.
        await env.run()


if __name__ == "__main__":
    fire.Fire(main)