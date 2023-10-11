from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
import os
import json
from pathlib import Path
from json import JSONDecodeError
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat_media import message
import os,re
from g4f import Provider, models
from langchain.llms.base import LLM

from langchain_g4f import G4FLLM
from CamelAgent import CamelAgent
from system_prompts import assistant1_inception_prompt,assistant2_inception_prompt,user_inception_prompt

icon = "🚀"
title="ThinkTankGPT"
st.set_page_config(
    page_title=title,
    page_icon=icon,
    layout="wide",
    menu_items={
        "Get help": "mailto:K00B404@gmail.com",
        "Report a bug": "mailto:GoldenKooy@gmail.com",
        "About": f"# *{icon}  ThinkTankGPT(powered by:HuggingFace)  {icon}* ",
    },
)


st.markdown("<style> iframe > div {    font-size: 24px;text-align: left;} </style>", unsafe_allow_html=True)
st.markdown("<div style='font-size: 24px;'> </div", unsafe_allow_html=True)
st.markdown("<div class='code-block'> </div>", unsafe_allow_html=True)

#stylesheet
#@st.cache(allow_output_mutation=True)
#def load_css(file_path):
 #   with open(file_path) as f:
 #       return f"<style>{f.read()}</style>"

#css = load_css("Camel_styles.css")

#st.markdown(css, unsafe_allow_html=True)

#  prompt snippets
prompt_snippets_save_file = 'Camel_prompt_snippet.txt'
prompt_snippet_text = ""

if os.path.exists(prompt_snippets_save_file):
    with open(prompt_snippets_save_file, "r") as f:
        prompt_snippet_text = f.read()

with st.expander("Snippets"):
    prompt_snippet = st.text_area("Prompt Snippet", prompt_snippet_text)

    b_col1, b_col2 = st.columns(2)
    if b_col1.button("Save Prompt Snippet"):
        with open(prompt_snippets_save_file, "w") as f:
            f.write(prompt_snippet)
        st.success("Prompt snippet saved!")

    if b_col2.button("Copy Prompt Snippet"):
        st.write("Copy the following prompt snippet:")
        st.code(prompt_snippet)



col1, col2 = st.columns(2)
assistant1_role_name = col1.text_input("Assistant 1 Role", "Senior Programmer")
assistant2_role_name = col2.text_input("Assistant 2 Role", "Grapics Designer")

col3, col4 = st.columns(2)
user_role_name = col3.text_input("User Role", "Game Developer")
nr_of_tasks = col4.text_input(" ", "Game Developer")

task = st.text_area("Task", "Develop a advanced ui only trade market simulation game in csharp and monobehaviour")
word_limit= st.number_input("Word Limit", 10, 1500, 50)


load_dotenv()

user_llm_free = G4FLLM(model=models.gpt_35_turbo,provider=Provider.Vercel,)


# the start of task 
if st.button("Start Autonomus AI AGENT"):
    task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
    task_specifier_prompt = """Here is a task that {assistant1_role_name} and {assistant2_role_name} will help {user_role_name} to complete: {task}.
    Please make it more specific. Be creative and imaginative.
    Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
    task_specifier_template = HumanMessagePromptTemplate.from_template(
        template=task_specifier_prompt
    )

    task_specify_agent = CamelAgent(
        task_specifier_sys_msg, user_llm_free
    )
    task_specifier_msg = task_specifier_template.format_messages(
        assistant1_role_name=assistant1_role_name,
        assistant2_role_name=assistant2_role_name,
        user_role_name=user_role_name,
        task=task,
        word_limit=word_limit,
    )[0]

    specified_task_msg = task_specify_agent.step(task_specifier_msg)

    print(f"Specified task: {specified_task_msg}")
    message(
        f"Specified task: {specified_task_msg}",
        allow_html=True,
        key="specified_task",
        avatar_style="adventurer",
    )

    specified_task = specified_task_msg

    # messages.py
    from langchain.prompts.chat import (
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )


    #print(assistant1_inception_prompt)
    #print(assistant2_inception_prompt)
    #print(user_inception_prompt)

    def get_sys_msgs(assistant1_role_name: str, assistant2_role_name: str, user_role_name: str, task: str):
        assistant1_sys_template = SystemMessagePromptTemplate.from_template(
            template=assistant1_inception_prompt
        )
        assistant1_sys_msg = assistant1_sys_template.format_messages(
            assistant1_role_name=assistant1_role_name,
            user_role_name=user_role_name,
            task=task,
        )[0]
        assistant2_sys_template = SystemMessagePromptTemplate.from_template(
            template=assistant2_inception_prompt
        )
        assistant2_sys_msg = assistant2_sys_template.format_messages(
            assistant2_role_name=assistant2_role_name,
            user_role_name=user_role_name,
            task=task,
        )[0]

        user_sys_template = SystemMessagePromptTemplate.from_template(
            template=user_inception_prompt
        )
        user_sys_msg = user_sys_template.format_messages(
            assistant1_role_name=assistant1_role_name,
            assistant2_role_name=assistant2_role_name,
            user_role_name=user_role_name,
            task=task,
        )[0]

        return assistant1_sys_msg, assistant2_sys_msg, user_sys_msg

    # Detect code blocks in response
    def detect_code_blocks(response):
        code_blocks = re.findall(r"```(.*?)```", response, re.DOTALL)
        return code_blocks


    # define the role system messages

    # define the role system messages
    assistant1_sys_msg, assistant2_sys_msg, user_sys_msg = get_sys_msgs(
        assistant1_role_name, assistant2_role_name, user_role_name, specified_task
    )

    # AI ASSISTANT setup                           |-> add the agent LLM MODEL HERE <-|
    assistant1_agent = CamelAgent(assistant1_sys_msg, user_llm_free)

   # AI ASSISTANT setup                           |-> add the agent LLM MODEL HERE <-|
    assistant2_agent = CamelAgent(assistant2_sys_msg, user_llm_free)

    # AI USER setup                      |-> add the agent LLM MODEL HERE <-|
    user_agent = CamelAgent(user_sys_msg, user_llm_free)

    # Reset agents
    assistant1_agent.reset()
    assistant2_agent.reset()
    user_agent.reset()

    # Initialize chats
    assistant1_msg = HumanMessage(
        content=(
            f"{user_sys_msg}. "
            "Now start to give me introductions one by one. "
            "Only reply with Instruction and Input."
        )
    )
    # Initialize chats
    assistant2_msg = HumanMessage(
        content=(
            f"{user_sys_msg}. "
            "Now start to give me introductions one by one. "
            "Only reply with Instruction and Input."
        )
    )

    user_msg1 = HumanMessage(content=f"{assistant1_sys_msg.content}")
    user_msg2 = HumanMessage(content=f"{assistant2_sys_msg.content}")

    user_msg1 = assistant1_agent.step(user_msg1)
    user_msg2 = assistant2_agent.step(user_msg2)

    message(
        f"AI Assistant 1 ({assistant1_role_name}):\n\n{user_msg1}\n\n",
        is_user=False,
        allow_html=True,
        key="0_assistant1",
        avatar_style="pixel-art",
    )

    message(
        f"AI Assistant 2 ({assistant2_role_name}):\n\n{user_msg2}\n\n",
        is_user=False,
        allow_html=True,
        key="0_assistant2",
        avatar_style="pixel-art",
    )

    chat_turn_limit, n = 30, 0

    detected_code_blocks = []

    while n < chat_turn_limit:
        n += 1
        # user msgs
        user_ai_msg1 = user_agent.step(assistant1_msg)
        user_ai_msg2 = user_agent.step(assistant2_msg)

        user_msg1 = HumanMessage(content=user_ai_msg1)
        #detected_code_blocks.append(detect_code_blocks(user_msg1 .content))
        user_msg2 = HumanMessage(content=user_ai_msg2)
        #detected_code_blocks.append(detect_code_blocks(user_msg2.content))

        message(
            f"AI User ({user_role_name}):\n\n{user_msg1.content}\n\n",
            is_user=True,
            allow_html=True,
            key=str(n) + "_user1",
        )

        message(
            f"AI User ({user_role_name}):\n\n{user_msg2.content}\n\n",
            is_user=True,
            allow_html=True,
            key=str(n) + "_user2",
        )
        # assistant msgs
        assistant_ai_msg1 = assistant1_agent.step(user_msg1)
        assistant_ai_msg2 = assistant2_agent.step(user_msg2)

        assistant1_msg = HumanMessage(content=assistant_ai_msg1)
        detected_code_blocks.append(detect_code_blocks(assistant1_msg.content))
        assistant2_msg = HumanMessage(content=assistant_ai_msg2)
        detected_code_blocks.append(detect_code_blocks(assistant2_msg.content))

        message(
            f"AI Assistant 1 ({assistant1_role_name}):\n\n{assistant1_msg.content}\n\n",
            is_user=False,
            allow_html=True,
            key=str(n) + "_assistant1",
            avatar_style="pixel-art",
        )

        message(
            f"AI Assistant 2 ({assistant2_role_name}):\n\n{assistant2_msg.content}\n\n",
            is_user=False,
            allow_html=True,
            key=str(n) + "_assistant2",
            avatar_style="pixel-art",
        )

        # ... Inside the loop for displaying responses
        # for i, user_code_block in enumerate(detected_code_blocks):
        #    st.code(user_code_block)
        #    if st.button(f"Save code block {i + 1} to file",key=f"save_button_{i}"):
        #        filename = st.text_input("Enter filename:", key=f"filename_{i}")
        #        fileext = st.text_input("Enter fileExtension:", key="fileext")
        #        if st.button("Save",key=f"save_{i}"):
        #            with open(f"generated_files/{filename}.{fileext}", "w") as f:
        #                f.write(user_code_block)
        #            st.success(f"Code block {i + 1} saved as {filename}.{fileext}")
        

        if (
            "<CAMEL_TASK_DONE>" in user_msg1.content
            or "task completed" in user_msg1.content
        ) and (
            "<CAMEL_TASK_DONE>" in user_msg2.content
            or "task completed" in user_msg2.content
        ):
            message("Task completed!", allow_html=True, key="task_done")
            break

        if "Error" in user_msg1.content or "Error" in user_msg2.content:
            message("Task failed!", allow_html=True, key="task_failed")
            break

