import os, getpass



from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from langchain_core.tools import tool
import requests
import re
from serpapi import GoogleSearch
from langchain_community.tools.tavily_search import TavilySearchResults
from semantic_router.encoders import OpenAIEncoder
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import ToolCall, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import BaseMessage
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]
SERP_API_KEY = os.environ["SERP_API_KEY"]
LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]


# Configuramos el modelo de lenguaje con OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    # model="gpt-3.5-turbo-0125",  # Alternativa de modelo si se desea una versión más económica
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0
)


# configure client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

encoder = OpenAIEncoder(name="text-embedding-3-small")

index_name = "gpt-4o-research-agent"

# connect to index
index = pc.Index(index_name)



serpapi_params = {
    "engine": "google",
    "api_key": os.getenv("SERP_API_KEY") or getpass("SerpAPI key: ")
    }


class AgentState(TypedDict):
    input: str ## representa la entrada actual
    chat_history: list[BaseMessage] ## mantiene un registro de la conversación pasada
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add] ##  permite rastrear las acciones intermedias o previas del agente


system_prompt = """Eres un Investigador Senior, eres un excelente tomador de decisiones.
Ante la consulta del usuario debes decidir qué hacer con ella en función de la
lista de herramientas que se te proporcionan.

Si ves que se ha utilizado una herramienta (en el scratchpad) con un propósito particular de
consulta, NO utilices esa misma herramienta con la misma consulta nuevamente. Además, NO utilices
cualquier herramienta más de dos veces (es decir, si la herramienta aparece en el scratchpad dos veces,
no volver a usarla).

Debes intentar recopilar información de una amplia gama de fuentes antes de
proporcionar la respuesta al usuario. Una vez que hayas recopilado mucha información
para responder la pregunta del usuario (almacenada en el scratchpad) usa la herramienta final_answer."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("assistant", "scratchpad: {scratchpad}"),
])


@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: str,
    main_body: str,
    conclusion: str,
    sources: str
):
    """Devuelve una respuesta en lenguaje natural al usuario en forma de un informe
    de investigación. Este informe incluye varias secciones, que son:
    
    - `introduction`: un párrafo corto que introduce la pregunta del usuario y el
      tema que estamos investigando.
    - `research_steps`: una lista de puntos que explica los pasos que se tomaron
      para investigar el informe.
    - `main_body`: aquí se incluye la mayor parte de la información de alta calidad 
      y concisa que responde a la pregunta del usuario. Debe tener entre 3-4 párrafos.
    - `conclusion`: un párrafo breve que proporciona una conclusión concisa pero
      sofisticada sobre los hallazgos.
    - `sources`: una lista de fuentes detalladas en forma de puntos, que incluye 
      todas las referencias utilizadas durante el proceso de investigación.
    """
    
    # Si 'research_steps' es una lista, la convierte en una cadena formateada con viñetas.
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    
    # Si 'sources' es una lista, también la convierte en una cadena con viñetas.
    if type(sources) is list:
        sources = "\n".join([f"- {s}" for s in sources])
    
    return ""


@tool("rag_search")
def rag_search(query: str):
    """Busca información especializada en IA usando una consulta en lenguaje natural."""
    xq = encoder([query])
    xc = index.query(vector=xq, top_k=2, include_metadata=True)
    context_str = format_rag_contexts(xc["matches"])
    return context_str

@tool("rag_search_filter")
def rag_search_filter(query: str, arxiv_id: str):
    """Busca información en nuestra base de datos de ArXiv utilizando una consulta en lenguaje natural
    y un ID específico de ArXiv. Esto nos permite obtener más detalles sobre un artículo específico."""
    xq = encoder([query])
    xc = index.query(vector=xq, top_k=6, include_metadata=True, filter={"arxiv_id": arxiv_id})
    context_str = format_rag_contexts(xc["matches"])
    return context_str

# Definimos una función auxiliar llamada 'format_rag_contexts' que recibe una lista de coincidencias 'matches'.
# Esta función toma cada coincidencia y extrae datos clave, como el título, contenido, ID de ArXiv y referencias relacionadas.
# Luego, formatea esta información en un bloque de texto estructurado y lo agrega a una lista de contextos 'contexts'.
# Finalmente, convierte esta lista en una sola cadena de texto 'context_str', donde cada contexto está separado por "---".
def format_rag_contexts(matches: list):
    contexts = []
    for x in matches:
        text = (
            f"Title: {x['metadata']['title']}\n"
            f"Content: {x['metadata']['content']}\n"
            f"ArXiv ID: {x['metadata']['arxiv_id']}\n"
            f"Related Papers: {x['metadata']['references']}\n"
        )
        contexts.append(text)
    context_str = "\n---\n".join(contexts)
    return context_str



@tool("web_search")
def web_search(query: str):
    """Finds general knowledge information using Google search. Can also be used
    to augment more 'general' knowledge to a previous specialist query."""

    tavily_search = TavilySearchResults(max_results=5)
    search_docs = tavily_search.invoke(query)

    return search_docs

@tool("web_search_serp")
def web_search_serp(query: str):
    """Encuentra información de conocimientos generales mediante la búsqueda de Google. También se puede utilizar
    para ampliar conocimientos más "generales" a una consulta especializada previa."""
    search = GoogleSearch({
        **serpapi_params,
        "q": query,
        "num": 5 ## Número de resultados
    })

    results = search.get_dict()["organic_results"]

    contexts = "\n---\n".join(
        ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
    )
    
    return contexts



## Búsqueda de papers en ArXiv
@tool("fetch_arxiv")
def fetch_arxiv(arxiv_id: str):
    """Obtiene el abstract de un artículo de ArXiv dado el ID de arxiv. Útil para
    encontrar contexto de alto nivel sobre un artículo específico."""

    ## Regex del abstract
    abstract_pattern = re.compile(
        r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
        re.DOTALL
    )

    # trae la pagina del paper en html
    res = requests.get(
        f"https://export.arxiv.org/abs/{arxiv_id}"
    )

    # busca el html del abstract
    re_match = abstract_pattern.search(res.text)

    # retorna el abstract del articulo
    return re_match.group(1)


# Lista de herramientas que el modelo podrá utilizar
tools = [
    rag_search_filter,
    rag_search,
    fetch_arxiv,
    web_search,
    web_search_serp,
    final_answer
]

## Función para transformar los pasos intermedios en un "scratchpad" de texto
def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":  # Solo agrega los pasos que ya se han ejecutado
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)

# Configuración del flujo de datos para el agente de IA, incluyendo el uso de herramientas
researcher = (
    {
        # Extrae el valor de entrada (input) desde el diccionario 'x'.
        # Esto se usará como la consulta inicial que recibe el agente de IA.
        "input": lambda x: x["input"],

        # Extrae el historial de chat desde 'x', permitiendo que el modelo mantenga contexto.
        "chat_history": lambda x: x["chat_history"],

        # Usa la función 'create_scratchpad' para procesar los pasos intermedios 
        # de investigación ('intermediate_steps') y los convierte en un "scratchpad".
        # Esto sirve como un registro de todas las herramientas usadas y sus resultados,
        # permitiendo al modelo a comprender el proceso previo y evitar repeticiones.
        "scratchpad": lambda x: create_scratchpad(
            intermediate_steps=x["intermediate_steps"]
        ),
    }
    # Se combina con el 'prompt'. Esto añade instrucciones al agente.
    | prompt

    # 'bind_tools' conecta las herramientas listadas en 'tools' al modelo (LLM),
    # permitiéndole elegir y utilizar cualquier herramienta que necesite para completar su tarea.
    | llm.bind_tools(tools, tool_choice="any")
)


def run_researcher(state: list):
    # Imprime el mensaje inicial y el contenido actual de 'intermediate_steps' en el estado
    print("run_researcher")
    print(f"intermediate_steps: {state['intermediate_steps']}")

    # Ejecuta el agente 'Researcher' con el estado actual como parámetro
    output = researcher.invoke(state)

    # Extrae el nombre y argumentos de la herramienta usada en 'output'
    tool_name = output.tool_calls[0]["name"]
    tool_args = output.tool_calls[0]["args"]

    # Crea un objeto 'AgentAction' con la herramienta, su entrada y un log pendiente ("TBD")
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="TBD"
    )

    # Devuelve el estado actualizado con el nuevo paso intermedio
    return {
        "intermediate_steps": [action_out]
    }

def router(state: list):
    # Determina el nombre de la herramienta a usar según el último paso en 'intermediate_steps'
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    else:
        # Si el formato es inválido, redirige a la herramienta "final_answer"
        print("Router invalid format")
        return "final_answer"
    
# Diccionario que asigna nombres de herramientas a sus funciones correspondientes
tool_str_to_func = {    
    "rag_search_filter": rag_search_filter,
    "rag_search": rag_search,
    "fetch_arxiv": fetch_arxiv,
    "web_search": web_search,
    "web_search_serp": web_search_serp,
    "final_answer": final_answer
}

def run_tool(state: list):
    # Obtiene el nombre de la herramienta y sus argumentos desde el último paso en 'intermediate_steps'
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    print(f"{tool_name}.invoke(input={tool_args})")  # Muestra la llamada de la herramienta con sus argumentos
    
    # Ejecuta la herramienta correspondiente usando el diccionario 'tool_str_to_func'
    output = tool_str_to_func[tool_name].invoke(input=tool_args)
    
    # Crea un nuevo objeto 'AgentAction' con la herramienta, su entrada, y el resultado obtenido
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(output)
    )
    
    # Devuelve el estado actualizado con el nuevo paso intermedio que incluye el resultado
    return {"intermediate_steps": [action_out]}


# Crea un grafo de estado (StateGraph) que maneja el estado del agente
graph = StateGraph(AgentState)

# Añade nodos al grafo, cada uno representando una herramienta o función
graph.add_node("researcher", run_researcher)
graph.add_node("rag_search_filter", run_tool)
graph.add_node("rag_search", run_tool)
graph.add_node("fetch_arxiv", run_tool)
graph.add_node("web_search", run_tool)
graph.add_node("web_search_serp", run_tool)
graph.add_node("final_answer", run_tool)

# Define el punto de entrada del grafo, donde comienza la ejecución
graph.set_entry_point("researcher")

# Agrega edges condicionales para decidir qué nodo seguir desde el "Researcher"
# Utiliza la función 'router' para determinar cuál es el próximo nodo a ejecutar
graph.add_conditional_edges(
    source="researcher",
    path=router,
)

# Crea conexiones de cada herramienta de vuelta al "Researcher", excepto "final_answer"
for tool_obj in tools:
    if tool_obj.name != "final_answer":
        graph.add_edge(tool_obj.name, "researcher")

# Si se llega a "final_answer", el flujo del grafo se dirige al nodo final 'END'
graph.add_edge("final_answer", END)

# Compila el grafo para hacerlo ejecutable
runnable = graph.compile()