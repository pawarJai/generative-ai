"""LangChain tool-calling agent with web search, weather, and news capabilities."""
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.config import llm
from app.models import QueryPlan
from app import state
from app.export.exporters import export_csv, export_excel, export_docx, export_pptx, export_chart
import requests
from datetime import datetime
import json


@tool
def make_csv(filename: str) -> str:
    """Export all extracted tables to a CSV file with the given filename."""
    return export_csv(state.get_active_file_id(), QueryPlan(intent="export", filename=filename))


@tool
def make_excel(filename: str) -> str:
    """Export all extracted tables to an Excel file with the given filename."""
    return export_excel(state.get_active_file_id(), QueryPlan(intent="export", filename=filename))


@tool
def make_docx(filename: str) -> str:
    """Export all extracted tables into a Word document with the given filename."""
    return export_docx(state.get_active_file_id(), QueryPlan(intent="export", filename=filename))


@tool
def make_pptx(filename: str) -> str:
    """Export all extracted tables as slides in a PowerPoint file with the given filename."""
    return export_pptx(state.get_active_file_id(), QueryPlan(intent="export", filename=filename))


@tool
def make_chart(filename: str) -> str:
    """Save a chart or table image (PNG) of the extracted data with the given filename."""
    return export_chart(state.get_active_file_id(), QueryPlan(intent="export", filename=filename))


@tool
def web_search(query: str) -> str:
    """Search the internet for information. Use for current events, news, general knowledge."""
    try:
        url = "https://duckduckgo.com/"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        params = {'q': query, 'format': 'json'}
        response = requests.get('https://api.duckduckgo.com/', params=params, headers=headers, timeout=5)
        data = response.json()

        results = []
        if data.get('AbstractText'):
            results.append(f"Summary: {data['AbstractText']}")
        if data.get('RelatedTopics'):
            for topic in data['RelatedTopics'][:3]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append(topic['Text'])

        return '\n'.join(results) if results else "No results found for: " + query
    except Exception as e:
        return f"Search failed: {str(e)}"


@tool
def get_weather(location: str) -> str:
    """Get current weather information for a location. Returns temperature, conditions, humidity."""
    try:
        url = f"https://wttr.in/{location}?format=j1"
        response = requests.get(url, timeout=5)
        data = response.json()

        current = data['current_condition'][0]
        temp_c = current['temp_C']
        temp_f = current['temp_F']
        desc = current['weatherDesc'][0]['value']
        humidity = current['humidity']
        wind_kph = current['windspeedKmph']

        return f"""Weather in {location}:
Temperature: {temp_c}°C ({temp_f}°F)
Condition: {desc}
Humidity: {humidity}%
Wind: {wind_kph} km/h"""
    except Exception as e:
        return f"Weather lookup failed for {location}: {str(e)}"


@tool
def get_news(topic: str) -> str:
    """Get latest news and headlines about a topic or location."""
    try:
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': topic,
            'sortBy': 'publishedAt',
            'pageSize': 5,
            'language': 'en'
        }
        # Try with or without API key
        response = requests.get(url, params=params, timeout=5)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])
            if articles:
                news = []
                for article in articles[:3]:
                    news.append(f"• {article['title']}\n  Source: {article['source']['name']}")
                return '\n\n'.join(news)

        # Fallback: search for news using web search
        search_query = f"latest news {topic}"
        return web_search(search_query)
    except Exception as e:
        return f"News lookup failed: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Calculation failed: {str(e)}"


@tool
def generate_csv_file(filename: str, data_description: str) -> str:
    """Generate and save a CSV file based on a description. Use this when creating files without uploaded documents."""
    try:
        import csv
        import os
        from app.config import OUTPUT_DIR

        # Create the CSV file directly
        filepath = os.path.join(OUTPUT_DIR, filename if filename.endswith('.csv') else f"{filename}.csv")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Ask LLM to generate CSV content
        prompt = f"""Generate CSV content for: {data_description}

Return ONLY valid CSV format with headers and data rows. No explanation, no markdown, no extra text.
Make it realistic and complete."""

        csv_content = llm.invoke(prompt).content.strip()

        # Clean up markdown fences if present
        csv_content = csv_content.replace('```csv', '').replace('```', '').strip()

        # Write to file
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            f.write(csv_content)

        # Count rows
        lines = csv_content.strip().split('\n')
        row_count = len(lines) - 1  # Subtract header

        return f"✓ Generated {filename} with {row_count} rows. Saved to {filepath}"
    except Exception as e:
        return f"CSV generation failed: {str(e)}"


@tool
def generate_excel_file(filename: str, data_description: str) -> str:
    """Generate and save an Excel file based on a description."""
    try:
        import os
        from app.config import OUTPUT_DIR

        filepath = os.path.join(OUTPUT_DIR, filename if filename.endswith('.xlsx') else f"{filename}.xlsx")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # First generate CSV
        prompt = f"""Generate CSV content for: {data_description}

Return ONLY valid CSV format with headers and data rows."""

        csv_content = llm.invoke(prompt).content.strip()
        csv_content = csv_content.replace('```csv', '').replace('```', '').strip()

        # Convert to Excel
        import pandas as pd
        from io import StringIO

        df = pd.read_csv(StringIO(csv_content))
        df.to_excel(filepath, index=False, sheet_name='Data')

        return f"✓ Generated {filename} with {len(df)} rows. Saved to {filepath}"
    except Exception as e:
        return f"Excel generation failed: {str(e)}"


_fallback_tools = [make_csv, make_excel, make_docx, make_pptx, make_chart,
                   web_search, get_weather, get_news, calculate,
                   generate_csv_file, generate_excel_file]
_agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You handle multi-step document requests. Use tools to produce every "
               "file the user asked for. Make at most one tool call per file requested."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])
_agent = create_tool_calling_agent(llm, _fallback_tools, _agent_prompt)
agent_executor = AgentExecutor(agent=_agent, tools=_fallback_tools, verbose=True,
                                max_iterations=10, handle_parsing_errors=True)
