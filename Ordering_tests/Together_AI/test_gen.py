import pandas as pd
from QATCH.qatch.connectors.sqlite_connector import SqliteConnector
from QATCH.qatch.generate_dataset.orchestrator_generator import OrchestratorGenerator

db_path = 'data/db_music.sqlite'
out_path = 'tests/music.csv'

# connection to the database
connector = SqliteConnector(
    relative_db_path= db_path,
    db_name='db'

)

# init the orchestrator
orchestrator_generator = OrchestratorGenerator()

# test generation
df: pd.DataFrame = orchestrator_generator.generate_dataset(connector)


#print(df.columns)

df_orderby = df


df.to_csv(out_path, index=False)

print(f'test generati in {out_path}')

