from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from typing import List
class CamelAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: None,
    ) -> None:
        self.system_message = system_message.content
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages
    
    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        try:
            messages = self.update_messages(input_message)
            output_message = self.model(str(input_message.content))
            self.update_messages(output_message)
            print(f"AI Assistant:\n\n{output_message}\n\n")
            return output_message
        except Exception as e:
            error_msg = f"Error occurred: {str(e)}"
            return AIMessage(content=error_msg)
