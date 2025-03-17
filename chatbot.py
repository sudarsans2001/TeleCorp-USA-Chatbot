# Import libraries
import pdfplumber
import pandas as pd
from datetime import datetime
import time
from langgraph.graph import StateGraph, END
from langchain.docstore.document import Document as LangChainDoc
import json
from typing import Dict, Any, List
import re
import logging
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer
import torch
from sentence_transformers import util
import openai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your OpenAI API key here
openai.api_key = ""  # Replace with your actual OpenAI API key

# State definition for LangGraph
class ChatbotState(Dict[str, Any]):
    query: str
    csv_data: pd.DataFrame
    tech_docs: List[LangChainDoc]
    vector_store: Any
    response: str

# MarketingDataAgent (Unchanged)
class MarketingDataAgent:
    def __init__(self, csv_path: str = "/content/telecom.csv"):
        self.data = pd.read_csv(csv_path)
        self.data.columns = self.data.columns.str.strip().str.lower()
        self.normalized_columns = {col.replace('_', ' ').lower(): col for col in self.data.columns if col != 'date'}
        self.data['date'] = pd.to_datetime(self.data['date'], format='%Y-%m-%d', errors='coerce')
        self.max_year = self.data['date'].dt.year.max()
        print("Marketing Data Agent initialized.")
        print("Columns:", list(self.data.columns))

    def _match_columns(self, query: str) -> List[str]:
        query_lower = query.lower().replace(' ', '_')
        query_words = set(query.lower().split())
        matched_cols = []
        for col_space, col_underscore in self.normalized_columns.items():
            col_space_lower = col_space.lower()
            col_underscore_lower = col_underscore.lower()
            col_words = set(col_space_lower.split())
            if (col_space_lower in query.lower() or 
                col_underscore_lower in query_lower or 
                col_space_lower.rstrip('s') in query.lower() or 
                col_space_lower + 's' in query.lower()):
                score = len(col_words.intersection(query_words)) / len(col_words)
                matched_cols.append((col_underscore, score))
        if not matched_cols:
            for col_space, col_underscore in self.normalized_columns.items():
                col_words = set(col_space.lower().split())
                if any(get_close_matches(word, col_words.union([col_underscore.lower()]), n=1, cutoff=0.85) for word in query_words):
                    score = len(col_words.intersection(query_words)) / len(col_words)
                    matched_cols.append((col_underscore, score))
        if matched_cols:
            matched_cols.sort(key=lambda x: x[1], reverse=True)
            return [matched_cols[0][0]]
        return []

    def process_query(self, state: ChatbotState) -> ChatbotState:
        query = state["query"].lower().strip()
        data = self.data
        logger.info(f"Marketing Agent processing query: {query}")

        date_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})', query)
        year_match = re.search(r'(?:in|for)\s+year\s+(\d{4})', query)
        month_match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', query)

        columns = self._match_columns(query)
        if not columns:
            state["response"] = "No relevant columns found in telecom.csv for this query."
            return state

        numeric_columns = [col for col in columns if pd.api.types.is_numeric_dtype(data[col])]
        if not numeric_columns:
            state["response"] = "Matched columns are not numeric and cannot be aggregated."
            return state

        if date_match and not (year_match or month_match):
            date_str = date_match.group(0)
            try:
                date = pd.to_datetime(date_str, errors='coerce')
                if pd.isna(date):
                    raise ValueError("Invalid date format")
                if date.year > self.max_year:
                    state["response"] = f"No data available for {date.strftime('%Y-%m-%d')} (data up to {self.max_year})."
                    return state
                result = data[data['date'] == date]
                if not result.empty:
                    results = {col: result[col].iloc[0] for col in numeric_columns}
                    state["response"] = " ".join(f"{col.replace('_', ' ')} was {val:,.2f}" 
                                                for col, val in results.items()) + f" on {date.strftime('%Y-%m-%d')}."
                else:
                    state["response"] = f"No data available for {date.strftime('%Y-%m-%d')} in {numeric_columns[0]}."
            except ValueError:
                state["response"] = "Could not parse date. Use DD-MM-YYYY or YYYY-MM-DD."
            return state

        filtered_data = data
        if year_match:
            year = int(year_match.group(1))
            if year > self.max_year:
                state["response"] = f"No data available for {year} (data up to {self.max_year})."
                return state
            filtered_data = data[data['date'].dt.year == year]
        elif month_match:
            month_str, year = month_match.groups()
            month = pd.to_datetime(month_str, format='%B').month
            year = int(year)
            if year > self.max_year:
                state["response"] = f"No data available for {month_str} {year} (data up to {self.max_year})."
                return state
            filtered_data = data[(data['date'].dt.year == year) & (data['date'].dt.month == month)]

        if filtered_data.empty:
            state["response"] = f"No data available for the specified period in {numeric_columns[0]}."
            return state

        agg_funcs = {
            "total": "sum",
            "max": "max",
            "maximum": "max",
            "min": "min",
            "minimum": "min",
            "avg": "mean",
            "average": "mean",
            "mean": "mean"
        }
        query_words = query.split()
        agg_type = None
        for key in agg_funcs:
            if key in query_words:
                agg_type = key
                break
        if not agg_type:
            if any("rate" in col for col in numeric_columns):
                agg_type = "average"
            else:
                agg_type = "total"

        results = {}
        for col in numeric_columns:
            result = getattr(filtered_data[col], agg_funcs[agg_type])()
            results[col] = 0 if pd.isna(result) else result

        agg_display = "Total" if agg_type == "total" else agg_type.capitalize()
        context = (f"in {year_match.group(1)} " if year_match else 
                   f"in {month_match.group(1)} {month_match.group(2)} " if month_match else "")
        state["response"] = " ".join(f"{agg_display} {col.replace('_', ' ')} {context}was {val:,.2f}" 
                                    for col, val in results.items()) + "."
        return state

# TechnicalSupportAgent (Hybrid: Exact match + Semantic search)
class TechnicalSupportAgent:
    def __init__(self, pdf_path: str = "/content/Error Codes.pdf"):
        """
        Initialize the TechnicalSupportAgent with a PDF file containing error codes.
        Extracts text, chunks it, and creates embeddings for RAG.
        """
        try:
            # Step 1: Extract text from PDF
            self.entries = self._extract_from_pdf(pdf_path)
            if not self.entries:
                raise ValueError("No error code entries extracted from PDF. Check file path or content structure.")

            # Step 2: Prepare corpus for semantic search
            self.corpus = [
                f"{code}: {details['Meaning']} {details['Cause']} {details['Resolution']}"
                for code, details in self.entries.items()
            ]
            self.corpus_keys = list(self.entries.keys())

            # Step 3: Create embeddings for RAG
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.corpus_embeddings = self.embedder.encode(self.corpus, convert_to_tensor=True)

            logger.info("Technical Support Agent initialized with PDF processing.")
            logger.info("Sample entries: %s", {k: v for k, v in list(self.entries.items())[:3]})

        except FileNotFoundError:
            logger.error(f"PDF file not found: {pdf_path}")
            raise
        except Exception as e:
            logger.error(f"Error initializing TechnicalSupportAgent: {str(e)}")
            raise

    def _extract_from_pdf(self, pdf_path: str) -> Dict[str, Dict[str, str]]:
        """
        Extract error code entries from a PDF file with '-' as field markers.
        Handles multi-line fields and stops at 'Common Issues'.
        """
        entries = {}
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                extracted = page.extract_text() or ""
                full_text += extracted + "\n"
                logger.debug(f"Extracted text from page: {extracted}")

            if not full_text.strip():
                logger.warning("No text extracted from PDF. Is it image-based or empty?")
                return entries

            logger.info(f"Full extracted text:\n{full_text}")

            lines = full_text.split('\n')
            current_code = None
            current_details = {"Meaning": "", "Cause": "", "Resolution": ""}
            current_field = None
            in_common_issues = False

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Stop processing at "Common Issues"
                if "Common Issues" in line:
                    in_common_issues = True
                    if current_code and any(current_details.values()):
                        entries[current_code] = current_details.copy()
                    current_code = None
                    continue
                if in_common_issues:
                    continue

                # Match error codes (standalone numbers or codes like "1228/875")
                code_match = re.match(r'^([0-9]+[a-zA-Z]*(?:[/-][0-9]+)?)$', line)
                if code_match:
                    # Save previous entry if it exists
                    if current_code and any(current_details.values()):
                        entries[current_code] = current_details.copy()
                    # Start new entry
                    current_code = code_match.group(1)
                    current_details = {"Meaning": "", "Cause": "", "Resolution": ""}
                    current_field = None
                    continue

                # Process fields if we have a current code
                if current_code:
                    if line.startswith("- Meaning"):
                        current_field = "Meaning"
                        current_details["Meaning"] = line[len("- Meaning"):].strip()
                    elif line.startswith("- Cause"):
                        current_field = "Cause"
                        current_details["Cause"] = line[len("- Cause"):].strip()
                    elif line.startswith("- Recommended Resolution") or line.startswith("- Recommended Resolutions"):
                        current_field = "Resolution"
                        current_details["Resolution"] = line[len("- Recommended Resolution"):].strip() if line.startswith("- Recommended Resolution") else line[len("- Recommended Resolutions"):].strip()
                    elif line.startswith("-") and current_field:
                        # Append multi-line content to the current field
                        current_details[current_field] += " " + line[1:].strip()
                    elif line.startswith("-") and not current_field:
                        # Handle cases where a line starts with '-' but no field is set (e.g., standalone sub-points)
                        if current_details["Resolution"]:
                            current_field = "Resolution"
                            current_details["Resolution"] += " " + line[1:].strip()
                        elif current_details["Cause"]:
                            current_field = "Cause"
                            current_details["Cause"] += " " + line[1:].strip()
                        elif current_details["Meaning"]:
                            current_field = "Meaning"
                            current_details["Meaning"] += " " + line[1:].strip()

            # Save the last entry
            if current_code and any(current_details.values()):
                entries[current_code] = current_details

        # Clean up and log results
        entries = {k: v for k, v in entries.items() if any(v.values())}
        if not entries:
            logger.warning("No structured error code entries found in PDF.")
        else:
            logger.info(f"Extracted {len(entries)} error code entries.")
        return entries

    def process_query(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user query by exact match or semantic search.
        """
        query = state["query"].lower().strip()
        logger.info(f"Technical Agent processing query: {query}")

        # Extract code from query
        code_match = re.search(r'(?:error\s*(?:code\s*)?|code\s*)(\d+[a-zA-Z]*(?:[/-]\d+)?)', query)
        if code_match:
            code = code_match.group(1)
            if code in self.entries:
                details = self.entries[code]
                if "resolution" in query:
                    state["response"] = f"Error {code}:\nResolution: {details['Resolution'].strip()}"
                else:
                    state["response"] = (
                        f"Error {code}:\n"
                        f"Meaning: {details['Meaning'].strip() or 'Not specified'}\n"
                        f"Cause: {details['Cause'].strip() or 'Not specified'}\n"
                        f"Resolution: {details['Resolution'].strip() or 'Not specified'}"
                    )
                return state

        # Fallback to semantic search
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=1)[0]
        corpus_id = hits[0]['corpus_id']
        relevant_key = self.corpus_keys[corpus_id]
        details = self.entries[relevant_key]

        if "resolution" in query:
            state["response"] = f"Error {relevant_key}:\nResolution: {details['Resolution'].strip()}"
        else:
            state["response"] = (
                f"Error {relevant_key}:\n"
                f"Meaning: {details['Meaning'].strip() or 'Not specified'}\n"
                f"Cause: {details['Cause'].strip() or 'Not specified'}\n"
                f"Resolution: {details['Resolution'].strip() or 'Not specified'}"
            )
        return state
        

# GeneralKnowledgeAgent (Unchanged)
class GeneralKnowledgeAgent:
    def __init__(self):
        print("General Knowledge Agent initialized with ChatGPT 3.5 Turbo API.")

    def process_query(self, state: ChatbotState) -> ChatbotState:
        query = state["query"].lower().strip()
        logger.info(f"General Agent processing query: {query}")

        if "current date" in query or "today" in query and "date" in query:
            current_date = datetime.now().strftime("%B %d, %Y")
            if "time" in query:
                current_time = datetime.now().strftime("%I:%M %p PDT")
                state["response"] = f"The current date is {current_date} and the time is {current_time}."
            else:
                state["response"] = f"The current date is {current_date}."
            return state

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a factual assistant providing answers based on knowledge up to December 2024. "
                            "Simulate a web/X search if needed. Provide concise, accurate answers. "
                            "If the answer requires data beyond December 2024, say: "
                            "'I don’t have sufficient data to answer this precisely as of December 2024.' "
                            "Do not invent facts."
                        )
                    },
                    {"role": "user", "content": query}
                ],
                max_tokens=200,
                temperature=0.1
            )
            state["response"] = response.choices[0].message["content"].strip()
        except Exception as e:
            state["response"] = f"Error contacting ChatGPT API: {str(e)}"
        return state

# Router Node (Unchanged)
def route_query(state: ChatbotState) -> Dict[str, str]:
    query = state["query"].lower()
    csv_columns = state["csv_data"].columns

    if (re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})|year\s+\d{4}|(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}', query) or 
        any(col.lower() in query.split() or col.replace('_', ' ').lower() in query for col in csv_columns)):
        if "current date" not in query and "today" not in query:
            agent = "marketing"
        else:
            agent = "general"
    elif "error" in query or "code" in query:
        agent = "technical"
    else:
        agent = "general"
    
    logger.info(f"Query: '{query}' -> Routed to: '{agent}'")
    return {"next": agent}

# Build LangGraph Workflow
def build_workflow():
    marketing_agent = MarketingDataAgent()
    technical_agent = TechnicalSupportAgent(pdf_path="/content/Error Codes.pdf")
    general_agent = GeneralKnowledgeAgent()

    graph = StateGraph(ChatbotState)
    
    graph.add_node("router", route_query)
    graph.add_node("marketing", marketing_agent.process_query)
    graph.add_node("technical", technical_agent.process_query)
    graph.add_node("general", general_agent.process_query)

    graph.add_conditional_edges(
        "router",
        lambda state: state["next"],
        {
            "marketing": "marketing",
            "technical": "technical",
            "general": "general"
        }
    )
    graph.add_edge("marketing", END)
    graph.add_edge("technical", END)
    graph.add_edge("general", END)

    graph.set_entry_point("router")
    
    return graph.compile()

# Interactive Interface
def run_chatbot():
    print("Welcome to TeleCorp USA Chatbot!")
    print("Ask about marketing data (e.g., 'total corp google discover spend in year 2023'), technical issues (e.g., 'what is error code 002?'), or general questions (e.g., 'what is the current date and time?').")
    print("Type 'exit' to quit.\n")

    workflow = build_workflow()
    initial_state = ChatbotState(
        query="",
        csv_data=pd.read_csv("/content/telecom.csv"),
        tech_docs=[],
        vector_store=None,
        response=""
    )
    initial_state["csv_data"]["date"] = pd.to_datetime(initial_state["csv_data"]["date"], format='%Y-%m-%d', errors='coerce')

    while True:
        query = input("Enter your question: ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break

        initial_state["query"] = query
        start_time = time.time()
        result = workflow.invoke(initial_state)
        exec_time = time.time() - start_time
        print(f"\nResponse: {result['response']}")
        print(f"Execution Time: {exec_time:.3f} seconds\n")

if __name__ == "__main__":
    run_chatbot()