import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import os
import pandas as pd

class PostgresAgent:
    def __init__(self):
        load_dotenv()

        self.connect()

    def connect(self):
        self.connection = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", 5432)
        )
        self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        
        # Set the schema (search_path)
        schema = os.getenv("POSTGRES_SCHEMA", "public")
        self.cursor.execute(f"SET search_path TO {schema};")
        
        
        
    def run_query(self, query, params=None):
        # Check if the query is read-only
        if not query.strip().lower().startswith(("select", "with")):
            return {"error": "Only read-only queries are allowed"}
        
        if self.connection.closed:
            self.connect()
        try:
            # check if result length is greater than 1'000'000
            subquery = query.rstrip(";")
            
            self.cursor.execute(f"SELECT COUNT(*) FROM ({subquery}) AS subquery;", params)
            result_count = self.cursor.fetchone()["count"]  
            if result_count > 1000000:
                limit_query = f"SELECT * FROM ({subquery}) AS subquery ORDER BY 1 LIMIT 10000"
                self.cursor.execute(limit_query, params)
                col_ordered_by = self.cursor.description[0][0]
                return {'result_length': result_count, f'result_1000000': self.cursor.fetchall(), 'col_ordered_by': col_ordered_by}
            else:
                self.cursor.execute(query, params)
                if self.cursor.description:
                    return {'result_length': result_count, f'result': self.cursor.fetchall(), 'col_ordered_by': None}
                
        except Exception as e:
            return {"error": str(e)}

    def run_query_df(self, query, params=None):
        res = self.run_query(query, params)
        if "error" in res:
            return res
        else:
            return pd.DataFrame(res)
        
        
    def close(self):
        self.cursor.close()
        self.connection.close()


if __name__ == "__main__":
    agent = PostgresAgent()
    
    # return as dictionary..
    # result = agent.run_query("""
    #     SELECT tablename
    #     FROM pg_catalog.pg_tables
    #     WHERE schemaname = current_schema();
    # """)
    # print(result)
    # agent.close()
    
    # # return as dataframe..
    result = agent.run_query_df("""
        SELECT *
        FROM biomarker_edrn
        LIMIT 10;
    """)
    print(result)
    agent.close()
    
    
    # result = agent.run_query_df("""
    # SELECT 
    #     table_name, 
    #     column_name, 
    #     data_type 
    # FROM 
    #     information_schema.columns 
    # WHERE 
    #     table_schema = 'oncomx_v1_0_25' 
    # ORDER BY 
    #     table_name, ordinal_position;
    # """)
    
    
    # print(result)
    # agent.close()
    