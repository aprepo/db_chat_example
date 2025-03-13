import streamlit as st
import sqlite3
import psycopg2
from google.cloud import bigquery
import json
import os
import openai

# Load API key from Streamlit secrets
openai_client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

# File to store metadata
METADATA_FILE = "db_metadata.json"


# Function to fetch schema metadata
def get_postgres_metadata(conn_str):
    conn = psycopg2.connect(conn_str)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public';
    """)
    metadata = {}
    for table_name, column_name, data_type in cursor.fetchall():
        if table_name not in metadata:
            metadata[table_name] = []
        metadata[table_name].append(column_name)
    cursor.close()
    conn.close()
    return metadata


def get_sqlite_metadata(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    metadata = {}
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info({table_name});")
        metadata[table_name] = [col[1] for col in cursor.fetchall()]
    cursor.close()
    conn.close()
    return metadata


def get_bigquery_metadata(project_id):
    client = bigquery.Client(project=project_id)
    metadata = {}
    datasets = list(client.list_datasets())
    for dataset in datasets:
        tables = client.list_tables(dataset.dataset_id)
        for table in tables:
            table_ref = client.get_table(table.reference)
            metadata[table.table_id] = [schema_field.name for schema_field in table_ref.schema]
    return metadata


def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)


def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {}


def filter_relevant_tables(metadata, user_prompt):
    relevant_tables = {}
    keywords = user_prompt.lower().split()
    for table, columns in metadata.items():
        if isinstance(columns, list):  # Ensure columns is a list
            if any(keyword in table.lower() or any(keyword in col.lower() for col in columns) for keyword in keywords):
                relevant_tables[table] = columns
    return relevant_tables if relevant_tables else metadata  # Default to full schema if nothing matches


def generate_sql_queries(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system",
                   "content": "You are an SQL query generator. Return only valid SQL queries based on the provided schema and database type. Do not use any quotes, markdown, or formatting. If multiple queries are required, separate them with a special delimiter '##QUERY##'. The response should contain nothing but the SQL queries themselves."},
                  {"role": "user", "content": prompt}],
        temperature=0.0,
        stream=False
    )
    return response.choices[0].message.content.strip().split("##QUERY##")


def generate_response_explanation(user_prompt, query_results):
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system",
                   "content": "You are a helpful assistant. Answer the user's original question using the provided query results from multiple queries."},
                  {"role": "user", "content": f"Original question: {user_prompt}\nQuery results: {query_results}"}],
        stream=False
    )
    return response.choices[0].message.content.strip()


def execute_queries(db_type, conn_str, queries):
    results = []
    if db_type == "PostgreSQL":
        conn = psycopg2.connect(conn_str)
    elif db_type == "SQLite":
        conn = sqlite3.connect(conn_str)
    else:
        return "BigQuery execution not implemented"

    cursor = conn.cursor()
    for query in queries:
        cursor.execute(query)
        results.append(cursor.fetchall())

    cursor.close()
    conn.close()
    return results


def main():
    st.title("Chat with Your Database")

    db_type = st.selectbox("Select Database Type", ["PostgreSQL", "SQLite", "BigQuery"])
    metadata = {}

    if db_type == "PostgreSQL":
        conn_str = st.text_input("Enter PostgreSQL Connection String")
        if st.button("Fetch Schema"):
            metadata = get_postgres_metadata(conn_str)
            save_metadata(metadata)

    elif db_type == "SQLite":
        db_path = st.text_input("Enter SQLite Database Path")
        if st.button("Fetch Schema"):
            metadata = get_sqlite_metadata(db_path)
            save_metadata(metadata)

    elif db_type == "BigQuery":
        project_id = st.text_input("Enter BigQuery Project ID")
        if st.button("Fetch Schema"):
            metadata = get_bigquery_metadata(project_id)
            save_metadata(metadata)

    st.subheader("Database Metadata")
    metadata = load_metadata()
    st.json(metadata)

    st.subheader("Chat with Database")
    user_prompt = st.text_area("Enter your request:")

    if st.button("Generate Query and Fetch Data"):
        relevant_metadata = filter_relevant_tables(metadata, user_prompt)
        schema_info = json.dumps(relevant_metadata)
        llm_prompt = f"Here is the relevant database schema:\n{schema_info}\nDatabase type: {db_type}\nGenerate SQL queries that retrieve the necessary data: {user_prompt}"
        generated_queries = generate_sql_queries(llm_prompt)

        st.text_area("Generated SQL Queries:", "\n".join(generated_queries))

        query_results = execute_queries(db_type, conn_str if db_type != "SQLite" else db_path, generated_queries)

        final_response = generate_response_explanation(user_prompt, query_results)

        st.subheader("Response")
        st.write(final_response)


if __name__ == "__main__":
    main()