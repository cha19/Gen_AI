from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv, dotenv_values



def get_user_inputs():
    char_name = input("Enter the main character's name: ")
    setting = input("Enter the story setting: ")
    theme = input("Enter the story theme: ")
    api_key = os.getenv("open_api_key")
    return char_name, setting, theme, api_key

def generate_story(character_name, setting, theme, api_key):
    prompt_template = PromptTemplate(
        input_variables=["character_name", "setting", "theme"],
        template="Create a story with the following details:\n\n"
                 "Main Character: {character_name}\n"
                 "Setting: {setting}\n"
                 "Theme: {theme}\n\n"
                 "Story:\n\n"
    )

    # Initialize the language model with the provided API key
    model = OpenAI(temperature=0.7, openai_api_key=api_key)

    # Create the chain
    chain = LLMChain(llm=model, prompt=prompt_template)

    # Generate the story
    story = chain.run(character_name=character_name, setting=setting, theme=theme)

    return story

if __name__ == '__main__':
    load_dotenv()
    char_name, setting, theme, api_key = get_user_inputs()
    story = generate_story(character_name=char_name, setting=setting, theme=theme, api_key=api_key)

    print('\nGenerated Story:\n')
    print(story)

