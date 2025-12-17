import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Union


class MessageType(Enum):
    EXPERT = "expert"
    HUMAN = "human"
    ROLE_GENERATOR = "role_generator"
    REVISER = "reviser"
    SYSTEM = "system"
    DEFAULT = "default"

    @classmethod
    def get_available_types(cls):
        print(list(MessageType.__members__.keys()))

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self == other
        try:
            return self.value == other.value
        except:
            pass
        try:
            if isinstance(other, float) or isinstance(other, int):
                return self.value == other
            if isinstance(other, str):
                return self.value == other
        except Exception as e:
            print(f"exception: {e}")
            pass
        return NotImplemented


@dataclass
class BaseMessage:
    r"""
    Base class for message objects used MA system.

    Args:
        content (str): The content of the message.
    """
    content: str
    """The string contents of the message."""

    type: Union[str, MessageType]
    """Message type: human, AI, system."""

    additional_kwargs: dict = dataclasses.field(default_factory=dict)
    """Any additional information."""

    def create_new_instance(self, content: str) -> "BaseMessage":
        r"""Create a new instance of the :obj:`BaseMessage` with updated
        content.

        Args:
            content (str): The new content value.

        Returns:
            BaseMessage: The new instance of :obj:`BaseMessage`.
        """
        return self.__class__(content=content, type=self.type)

    def __add__(self, other: Any) -> Union["BaseMessage", Any]:
        r"""Addition operator override for :obj:`BaseMessage`.

        Args:
            other (Any): The value to be added with.

        Returns:
            Union[BaseMessage, Any]: The result of the addition.
        """
        if isinstance(other, BaseMessage):
            combined_content = self.content.__add__(other.content)
        elif isinstance(other, str):
            combined_content = self.content.__add__(other)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for +: '{type(self)}' and "
                f"'{type(other)}'")
        return self.create_new_instance(combined_content)

    def __mul__(self, other: Any) -> Union["BaseMessage", Any]:
        r"""Multiplication operator override for :obj:`BaseMessage`.

        Args:
            other (Any): The value to be multiplied with.

        Returns:
            Union[BaseMessage, Any]: The result of the multiplication.
        """
        if isinstance(other, int):
            multiplied_content = self.content.__mul__(other)
            return self.create_new_instance(multiplied_content)
        else:
            raise TypeError(
                f"Unsupported operand type(s) for *: '{type(self)}' and "
                f"'{type(other)}'")

    def __len__(self) -> int:
        r"""Length operator override for :obj:`BaseMessage`.

        Returns:
            int: The length of the content.
        """
        return len(self.content)

    def __contains__(self, item: str) -> bool:
        r"""Contains operator override for :obj:`BaseMessage`.

        Args:
            item (str): The item to check for containment.

        Returns:
            bool: :obj:`True` if the item is contained in the content,
                :obj:`False` otherwise.
        """
        return item in self.content

    def to_dict(self) -> Dict:
        r"""
        Converts the message to a dictionary.

        Returns:
            dict: The converted dictionary.
        """
        return {
            "content": self.content,
            "type": self.type,
            **(self.additional_kwargs or {})
        }