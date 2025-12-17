import copy
from typing import Union, Any

from evaluation_suit.messages import BaseMessage, MessageType


class HumanMessage(BaseMessage):
    """
    A Message from a human.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(type=MessageType.HUMAN,
                         **kwargs)


class SystemMessage(BaseMessage):
    """
    A Message for System role for an AI Agent.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(type=MessageType.SYSTEM,
                         **kwargs)
        self.initial_content: str = copy.deepcopy(self.content)

    def update_role(self, role) -> None:
        self.content = self.initial_content.format(role=role)


class ExpertMessage(BaseMessage):
    """
    A Message from an AI expert.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(type=MessageType.EXPERT,
                         **kwargs)


class RoleGeneratorMessage(BaseMessage):
    """
    A Message from a Role Generator AI Agent.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(type=MessageType.ROLE_GENERATOR,
                         **kwargs)


class ReviserMessage(BaseMessage):
    """
    A Message from a Reviser AI Agent.
    """
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(type=MessageType.REVISER,
                         **kwargs)
