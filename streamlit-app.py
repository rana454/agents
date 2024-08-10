import streamlit as st
import requests
import json
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import io
import requests, json

import os
import json

from pydantic import BaseModel

import os
import yaml
import pandas as pd
from crewai import Agent, Task, Crew, Process
from crewai_tools import CodeInterpreterTool, SerperDevTool
from crewai_tools import FileReadTool
from crewai_tools import tool
from pydantic import BaseModel
from dotenv import load_dotenv

# Define the function to create an Insight Aggregator Agent
def create_insight_aggregator_agent():
    return Agent(
        role="Insight Aggregator",
        goal="Combine insights from the Data Analytics crew and the PDF RAG system",
        backstory=(
            "You are an expert at synthesizing information only on given data."
            "Your role is to combine the data-driven insights with knowledge"
        ),
        allow_delegation=False,
        verbose=True,
        max_iter=2,
    )


# Define the function to create an Insight Aggregation Task
def create_insight_aggregation_task(agent):
    return Task(
        description=(
            "Receive the user query: {query} and the insights in form of the {dataframe},"
            "Combine all insights and present a comprehensive answer."
        ),
        expected_output="A detailed analytical data insights based on {dataframe}.",
        agent=agent,
        async_execution=False,
    )


# "A detailed response combining both data insights and document-based knowledge.
# Define the function to form the Insight Aggregation Crew
def create_insight_aggregation_crew(agent, task):
    return Crew(agents=[agent], tasks=[task], process=Process.sequential)


# Define the main function to execute the Insight Aggregation Crew
def execute_insight_aggregation(user_query, data_analysis_result):
    agent = create_insight_aggregator_agent()
    task = create_insight_aggregation_task(agent)
    crew = create_insight_aggregation_crew(agent, task)

    return crew.kickoff(
        inputs={
            "query": user_query,
            "dataframe": data_analysis_result,
        }
    )


# Example usage:
# final_insight = execute_insight_aggregation(user_query, data_analysis_result)

_ = load_dotenv()

os.environ["SERPER_API_KEY"] = st.secrets("SERPER_API_KEY")
os.environ["OPENAI_API_KEY"] = st.secrets("OPENAI_API_KEY")


# Load agents and tasks from YAML files
def load_config(filename):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


agents_config = load_config("config/agents.yaml")
tasks_config = load_config("config/tasks.yaml")


@tool
def read_csv_and_get_info(file_name: str):
    """
    Reads a CSV file and find the header and column info.

    Args:
    - file_name: The path to the CSV file.

    Returns:
    - The header and the column info  .
    """
    # csv_file = f"{classname}.csv"
    df = pd.read_csv(file_name)
    description = df.head(3).to_markdown()
    columns_info = df.dtypes.to_string()

    return description, columns_info


@tool
def read_and_execute(file_name: str):
    """
    Reads a CSV file and return the dataframe to execute the code.

    Args:
    - file_name: The path to the CSV file.

    Returns:
    - return the dataframe .
    """
    # csv_file = f"{classname}.csv"
    df = pd.read_csv(file_name)
    return df


@tool
def execute_code_tool(code: str, file_name: str) -> pd.DataFrame:
    """Tool to read a CSV file, execute Python code on the DataFrame, and return the modified DataFrame.
    Args:
    - file_name: The path to the CSV file.
    - code: The Python code to execute on the DataFrame.
    Returns:
    - The modified DataFrame.
    """
    import pandas as pd

    df = pd.read_csv(file_name)
    exec_globals = {"df": df}
    exec_locals = {}
    exec(code, exec_globals, exec_locals)
    return exec_globals["df"]


@tool
def find_unique_values_in_categorical_columns(file_name: str) -> str:
    """Tool to read a CSV file,and help the agents to make the sense of the user query by finding
    the unique values in each categorical column, and return a markdown string."""
    # Read the CSV file into a DataFrame
    import pandas as pd

    df = pd.read_csv(file_name)

    # Initialize a markdown string
    markdown_output = "# Unique Values in Categorical Columns\n\n"

    # Iterate through each column in the DataFrame
    for column in df.columns:
        # Check if the column is categorical (object or category type)
        if df[column].dtype == "object" or df[column].dtype.name == "category":
            unique_values = df[column].unique()
            if unique_values / df.shape[0] > 0.2:
                continue
            markdown_output += f"## {column}\n"
            markdown_output += ", ".join([str(value) for value in unique_values])
            markdown_output += "\n\n"

    return markdown_output


# Initialize agents
query_classification_agent = Agent(
    **agents_config["query_classification_agent"], verbose=True, max_iter=1
)

data_extraction_agent = Agent(
    **agents_config["data_extraction_agent"],
    tools=[read_csv_and_get_info, execute_code_tool],
    max_iter=2,
    verbose=True,
    allow_delegation=False,
)

visualization_agent = Agent(
    **agents_config["visualization_agent"],
    tools=[
        read_csv_and_get_info,
        execute_code_tool,
        find_unique_values_in_categorical_columns,
    ],
    max_iter=10,
    verbose=True,
    allow_delegation=False,
)

# Initialize tasks
query_classification_task = Task(
    description=tasks_config["query_classification_task"]["description"],
    expected_output=tasks_config["query_classification_task"]["expected_output"],
    agent=query_classification_agent,
)

data_extraction_task = Task(
    description=tasks_config["data_extraction_task"]["description"],
    expected_output=tasks_config["data_extraction_task"]["expected_output"],
    agent=data_extraction_agent,
)


class DataExtractionoutput(BaseModel):
    code: str
    dataframe: str
    explaination: str


visualization_task = Task(
    description=tasks_config["visualization_task"]["description"],
    expected_output=tasks_config["visualization_task"]["expected_output"],
    agent=visualization_agent,
    output_json=DataExtractionoutput,
)

# Form initial crew for query classification
initial_crew = Crew(
    agents=[query_classification_agent],
    tasks=[query_classification_task],
    process=Process.sequential,
    output_log_file=True,
)


def execute_initial_crew(inputs):
    # Kickoff the initial crew process
    classification_result = initial_crew.kickoff(inputs=inputs)

    # # # Extract class name from classification result
    class_name = classification_result
    if class_name == "Unknown":
        raise ValueError(
            "The query could not be classified. Please try again with a different query."
        )

    # Read CSV and get DataFrame info
    class_name = "Agricultural-Heatmaps"

    # df, header, dataframe_info = read_csv_and_get_info(class_name)

    # Update inputs with DataFrame info
    # updated_inputs = {
    #     'dataframe_info': dataframe_info,
    #     'header': header,
    #     'file_name': '/home/tagbin/Documents/BIPARD/crewai/Agricultural-Heatmaps_processed.csv',
    # }

    return class_name


def execute_data_extraction_and_visualization(inputs):
    # Form crew for data extraction and visualization

    data_extraction_crew = Crew(
        agents=[visualization_agent],
        tasks=[visualization_task],
        process=Process.sequential,
    )

    # Kickoff the data extraction and visualization process
    result = data_extraction_crew.kickoff(inputs=inputs)
    return result


class_descriptions = {
    "Agriculture_Heatmap": "Detailed agricultural data for Bihar, India, including focus areas, indicators, years, districts, values, units, and sources. Useful for in-depth analysis and cross-referencing.",
    "Agriculture_Trends": "Comprehensive data on agriculture and allied sectors in Bihar, including sector names, focus areas, indicators, types, years, values, units, and sources. Useful for analyzing temporal trends.",
    "Rural_Development": "Information on rural development initiatives in Bihar, categorized by sector, focus area, indicators, years, detailed steps, measurable outcomes, and sources. Useful for monitoring progress and policy-making.",
}


def classify_query(query: str):
    initial_inputs = {"query": query, "class_descriptions": class_descriptions}
    return execute_initial_crew(initial_inputs)


def get_data_analysis_result(query: str, file_name: str):
    inputs = {"query": query, "file_name": file_name}
    return execute_data_extraction_and_visualization(inputs)


def process_query(query: str):
    query_type = classify_query(query)
    query_type = query_type.strip()
    print(f"query_type {query_type}")
    file_path = "./data/Agricultural-Heatmaps_processed.csv"

    if query_type in [
        "Agriculture_Heatmaps",
        "Agriculture_Heatmap",
        "Agriculture-Heatmaps",
        "Agriculture-Heatmap",
    ]:
        file_path = "./data/Agricultural-Heatmaps_processed.csv"
    elif "Agriculture_Trends" in query_type:
        file_path = "./data/Agricultural-Heatmaps_processed.csv"
    elif "Rural_Development" in query_type:
        file_path = "./data/Rural-Development_processed.csv"
    print(f"file_path {file_path}")

    crew_output = get_data_analysis_result(query=query, file_name=file_path)
    # rag_result = send_query(query)
    rag_result = "Agricultural-Heatmaps data"

    analytics_data = crew_output.json_dict["dataframe"]
    explanation = execute_insight_aggregation(query, analytics_data)

    return {
        "query_type": query_type,
        "file_path": file_path,
        "crew_output": crew_output,
        "rag_result": rag_result,
        "explanation": explanation,
    }


st.set_page_config(layout="wide")
st.title("Data Analysis and Visualization Chat App")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create a container for chat messages
chat_container = st.container()


# Function to display all messages
def display_messages():
    for message in st.session_state.messages:
        with chat_container.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                if isinstance(message["content"], str):
                    try:
                        content = json.loads(message["content"])
                    except json.JSONDecodeError:
                        content = {"error": "Invalid JSON"}
                elif isinstance(message["content"], dict):
                    content = message["content"]
                else:
                    content = {"error": "Unknown content type"}

                st.write(f"File to be read: {content.get('file_path', 'N/A')}")

                if "crew_output" in content and "json_dict" in content["crew_output"]:
                    chart_code = content["crew_output"]["json_dict"].get("code")
                    if chart_code:
                        code_lines = chart_code.split("\n")
                        data_prep_code = "\n".join(
                            [
                                line
                                for line in code_lines
                                if "st.set_page_config" not in line
                            ]
                        )

                        # Execute data preparation and figure creation
                        exec_globals = {}
                        exec(data_prep_code, exec_globals, globals())

                        # Uncomment the following block if you want to display the chart and table
                        # if 'fig' in globals():
                        #     st.plotly_chart(fig, use_container_width=True)
                        #
                        #     # Find the last DataFrame created in the code
                        #     dataframes = [var for var in globals() if isinstance(globals()[var], pd.DataFrame)]
                        #     if dataframes:
                        #         table = globals()[dataframes[-1]]
                        #         st.dataframe(table)
                        #     else:
                        #         st.write("No data table available.")
                        # else:
                        #     st.write("Chart generation failed. Please check the code.")
                    else:
                        st.write("No chart code available")

                if "explanation" in content:
                    st.write("Explanation:")
                    st.write(content["explanation"].raw)


# Display existing messages
display_messages()

# Create a placeholder for new messages
message_placeholder = st.empty()

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Send user query to backend
    with st.spinner("Analyzing..."):
        process_response = process_query({"text": prompt})

    if process_response:
        result = process_response

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result})
    else:
        st.error("Error processing query")

    # Clear the placeholder and display all messages
    message_placeholder.empty()
    display_messages()

# Add JavaScript to scroll to bottom
st.markdown(
    """
<script>
    var body = window.parent.document.querySelector(".main");
    body.scrollTop = body.scrollHeight;
</script>
""",
    unsafe_allow_html=True,
)
