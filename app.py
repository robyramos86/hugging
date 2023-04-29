from glob import glob
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os


os.environ["OPENAI_API_KEY"] = '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'



def construct_index(directory_path):
    max_input_size = 3500
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index


def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')

    # Ler e concatenar os documentos da pasta "docs" como o contexto relevante
    docs_directory_path = "docs"
    documents = ""
    for file_path in glob(os.path.join(docs_directory_path, "*.{txt,pdf}")):
        with open(file_path, "r") as f:
            documents += f.read() + " "
    contexto = documents.strip()

    # Combinar o contexto e a pergunta de entrada
    with open('texto.txt', 'r') as f:
        texto_prefixo = f.readline().strip()
    texto_entrada = f"{texto_prefixo}{input_text}{contexto}"

      
    response = index.query(texto_entrada, response_mode="compact")
    return response.response


description = """
A IA foi treinada com materiais do Argon Fundamentals e do canal do Youtube do nosso treinador André Gomes para responder perguntas sobre agilidade. Faça sua pergunta!
"""

iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Como podemos te ajudar?"),
                     outputs="text",
                     description=description,                     
                     title="MentorIA Ágil GPT (Beta)")


index = construct_index("docs")
iface.launch(share=False)