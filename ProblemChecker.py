import re
from metagpt.actions import Action
from metagpt.roles import Role
import asyncio
from metagpt.context import Context
from metagpt.logs import logger
from metagpt.schema import Message

'''
and it is correctly stated with all its vertices, nodes and weights.

'''

class ProblemCheck(Action):
    PROMPT_TEMPLATE: str = """
    Given the following max-cut problem:
    {instruction}.
    Return yes if the provided problem is a max-cut problem with enough information to create a mathematical formulation.
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
        self.set_actions([ProblemCheck])

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
    role = ProblemCheck(context=context)
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)

asyncio.run(main())

