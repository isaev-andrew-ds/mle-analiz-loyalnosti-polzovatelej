from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy import Engine


class DataProcessor:

    def __init__(
        self,
        df_config: dict,
        df_query: str, 
        tenge_df_url: str
    ):
        self.df_config = df_config
        self.df_query = df_query
        self.engine = self._connect()
        self.df = self._get_df_from_sql_query(self.df_query, self.engine)
        self.tenge_df = self._get_df_from_request(self.tenge_df_url)

    def _connect(self) -> Engine:
        connection_string = "postgresql://{}:{}@{}:{}/{}".format(
            self.df_config["user"],
            self.df_config["pwd"],
            self.df_config["host"],
            self.df_config["port"],
            self.df_config["db"],
        )
        return create_engine(connection_string)

    def _get_df_from_sql_query(self, query: str, engine: Engine) -> pd.DataFrame:
        return pd.read_sql_query(query, con=engine)

    def _get_df_from_request(self, url: str) -> pd.DataFrame:
        return pd.read_csv(url)
    