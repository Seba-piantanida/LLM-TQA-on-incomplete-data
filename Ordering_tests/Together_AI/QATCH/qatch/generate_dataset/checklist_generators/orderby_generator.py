from QATCH.qatch.connectors import ConnectorTable
from .base_generator import BaseGenerator, SingleQA
from .utils import utils_list_sample
import random
from itertools import combinations, islice
import json


class OrderByGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'ORDERBY'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        num_cols = list(table.num_col2metadata.keys())
        columns = list(table.tbl_col2metadata.keys())
        
        tbl_name = table.tbl_name
        try:
            rem_columns = json.load(open('removable_colums.json'))[tbl_name]['removable_columns']
        
        except:
            rem_columns = []
        
        select_tests = self.generate_all_table_order(columns, tbl_name, rem_columns)
        select_tests += self.generate_advanced_order(num_cols, tbl_name, rem_columns)
        select_tests += self.generate_single_col_order(columns, tbl_name, rem_columns)
        

        return select_tests

    def generate_all_table_order(self, columns, tbl_name, rem_columns) -> list[SingleQA]:
        """
        This method generates SQL queries and respective questions for testing
        ORDER BY clause on every column of a table, in both ascending and
        descending order. It returns a list of SingleQA typed dictionaries,
        where each SingleQA dictionary contains an SQL query, its respective
        question and a test tag.

        Args:
            columns (List[str]): List of column names of the table.
            tbl_name (str): Name of the table.

        Returns:
            List[SingleQA]: A list of SingleQA typed dictionaries. Each
            SingleQA contains:
                query (str): The generated SQL query.
                question (str): The corresponding question of the SQL query.
                sql_tag (str): Tag representing the type of SQL query.

        Notes:
            - This method always uses a sample of only two columns from the
            provided list, even if more columns are provided to avoid explosion.
        """

        # number of tests: len(columns) * 2
        if len(rem_columns) > 0:
            columns = rem_columns

        columns = utils_list_sample(columns, k=5, val=self.column_to_include)

        tests = []
        operations = [
            ('ASC', 'ascending'),
            ('DESC', 'descending'),
        ]
        limit = 5 
        for col in columns:
            for operation in operations:
                single_qa = SingleQA(
                    query=f'SELECT * FROM `{tbl_name}` ORDER BY `{col}` {operation[0]} LIMIT {limit}',
                    question = f'List the top {limit} rows from the table `{tbl_name}`, sorted by the column `{col}` in {operation[1]} order.',
                    sql_tag='ORDERBY-SINGLE',
                )
                tests.append(single_qa)

        ####### old prompt more sintatticaly complex#####

        # for col in columns:
        #     for operation in operations:
        #         single_qa = SingleQA(
        #             query=f'SELECT * FROM `{tbl_name}` ORDER BY `{col}` {operation[0]} LIMIT {limit}',
        #             question=f'Show the first {limit} entries ordered by {col} in {operation[1]} order for the table {tbl_name}',
        #             sql_tag='ORDERBY-SINGLE',
        #         )
        #         tests.append(single_qa)
             
        return tests

    def generate_single_col_order(self, columns, tbl_name,rem_columns) -> list[SingleQA]:
        """
        Creates a list of `SingleQA` objects representing queries that select a random column from
        the provided columns and sort the selected column data in ascending or descending order.

        The function takes a list of columns and randomly selects one column using the `utils_list_sample` function.
        For the selected column, it creates queries to sort the data in both ascending (ASC) and descending (DESC) order.

        Notes:
            This function generates `len(columns) * 2` number of tests, as for each column two tests
            (ascending and descending order) are created. The column used for generating the tests is
            selected randomly from the provided list of columns.

        Args:
            columns (list): A list of column names for which the ascending and descending sort queries are to be created.
            tbl_name (str): The name of the table that contains the columns for which the queries will be generated.

        Returns:
            list[SingleQA]: A list of `SingleQA` objects representing the tests. Each `SingleQA` object contains
                             the query, the question to be asked i.e., the projection of the column in a particular
                             (ascending or descending) order, and the SQL tag for the test.

        """

        # number of tests: len(columns) * 2
        if len(rem_columns) > 0:
            columns = rem_columns
        columns = utils_list_sample(columns, k=5, val=self.column_to_include)

        tests = []
        operations = [
            ('ASC', 'ascending'),
            ('DESC', 'descending'),
        ]
        limit = 5
        for col in columns:
            for operation in operations:
                single_qa = SingleQA(
                    query=f'SELECT `{col}` FROM `{tbl_name}` ORDER BY `{col}` {operation[0]} LIMIT {limit}',
                    question = (
                                    f'Select only the column `{col}` from `{tbl_name}`. '
                                    f'Sort it in {operation[1]} order and return the first {limit} values.'
                                ),
                    sql_tag='ORDERBY-PROJECT',
                )
                tests.append(single_qa)

        ####### old prompt more sintatticaly complex#####

        # for col in columns:
        #     for operation in operations:
        #         single_qa = SingleQA(
        #             query=f'SELECT `{col}` FROM `{tbl_name}` ORDER BY `{col}` {operation[0]} LIMIT {limit}',
        #             question=f'Project the {col} ordered in {operation[1]} order for the table {tbl_name} and show only the first {limit} entries',
        #             sql_tag='ORDERBY-PROJECT',
        #         )
        #         tests.append(single_qa)


        return tests

    def generate_advanced_order(self, columns, tbl_name, rem_columns) -> list[SingleQA]:
        """
        This method generates SQL queries and respective questions for testing
        ORDER BY on sum value of various combinations of numerical columns in both ascending and
        descending order. It returns a list of SingleQA typed dictionaries,
        where each SingleQA dictionary contains an SQL query, its respective
        question and a test tag.

        Args:
            columns (List[str]): List of numerical column names of the table.
            tbl_name (str): Name of the table.

        Returns:
            List[SingleQA]: A list of SingleQA typed dictionaries. Each
            SingleQA contains:
                query (str): The generated SQL query.
                question (str): The corresponding question of the SQL query.
                sql_tag (str): Tag representing the type of SQL query.

        Notes:
            - This method always uses a sample of only two columns from the
            provided list, even if more columns are provided to avoid explosion.
        """
       
        tests = []
        operations = [
            ('ASC', 'ascending'),
            ('DESC', 'descending'),
        ]
        if len(rem_columns) > 0:
            columns = [c for c in columns if c in rem_columns]
        

        all_combinations = []
        k = 3
        n = 5
        for r in range(2, k + 1):
            all_combinations.extend(combinations(columns, r))
        random.shuffle(all_combinations)
        n = min(n, len(all_combinations))
        col_combinations = all_combinations[:n]

        limit = 5
        for i, col in enumerate(col_combinations):
            for operation in operations:
                single_qa = SingleQA(
                    query=f'SELECT * FROM `{tbl_name}` ORDER BY `{"` + `".join(col)}` {operation[0]} LIMIT {limit}',
                    question = (
                                    f'Compute the sum of the columns {", ".join(col)} for each row in `{tbl_name}`. '
                                    f'Sort all rows in {operation[1]} order by this sum and return the first {limit} rows.'
                                ),
                    sql_tag='ORDERBY-ADVANCED',
                )
                tests.append(single_qa)

         ####### old prompt more sintatticaly complex#####


        # for i, col in enumerate(col_combinations):
        #     for operation in operations:
        #         single_qa = SingleQA(
        #             query=f'SELECT * FROM `{tbl_name}` ORDER BY `{"` + `".join(col)}` {operation[0]} LIMIT {limit}',
        #             question=f'Show the first {limit} entries ordered by the sum of {" + ".join(col)}, in {operation[1]} order for the table {tbl_name}',
        #             sql_tag='ORDERBY-ADVANCED',
        #         )
        #         tests.append(single_qa)


        return tests