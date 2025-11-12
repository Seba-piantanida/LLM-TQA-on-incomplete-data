from QATCH.qatch.connectors import ConnectorTable
from .base_generator import BaseGenerator, SingleQA
from .utils import utils_list_sample
import random
from itertools import combinations, islice
import json

class JoinGenerator(BaseGenerator):
    @property
    def test_name(self):
        return 'INNER-JOIN'

    def template_generator(self, table: ConnectorTable) -> list[SingleQA]:
        """
        Generates templates for the given ConnectorTable. The templates are created by generating
        Join Project All tests and Join Project Single tests. If the ConnectorTable has no foreign
        keys, an empty list is returned.

        Args:
            table (ConnectorTable): The ConnectorTable object for which templates will be generated.

        Returns:
            list[SingleQA]: A list of SingleQA tests that have been generated. Each SingleQA is a dictionary
                            containing a SQL query, a corresponding natural language question, and a SQL tag.

        Note:
            The returned list of tests can have varying lengths depending on the foreign keys in the input
            ConnectorTable and the category column metadata of the parent and child tables involved in the
            foreign key relationships.
        """

        num_cols = list(table.num_col2metadata.keys())
        columns = list(table.tbl_col2metadata.keys())
        
        tbl_name = table.tbl_name
        try:
            rem_columns = json.load(open('removable_colums.json'))[tbl_name]['removable_columns']
        
        except:
            rem_columns = []

        if len(table.foreign_keys) == 0:
            return []
        tests = self.generate_join_project_all(table, columns, rem_columns)
        tests += self.generate_join_project_single(table, columns, rem_columns)
        tests += self.generate_advanced_join_order(columns=num_cols, rem_columns=rem_columns, tbl_name=tbl_name, table=table)
        return tests

    def generate_join_project_all(self, table: ConnectorTable, columns, rem_columns) -> list[SingleQA]:
        """
        Generates a list of SQL queries and questions representing all possible joins
        between a given table and all its child tables based on foreign key relationships.
        The join operation will use all the records from the given table and the child tables.

        Args:
            table (ConnectorTable): The table to generate join queries for.

        Returns:
            list[SingleQA]: List of dictionaries holding generated SQL queries and questions.
                            Each dictionary has the keys:
                                - 'query': The SQL query for the join operation.
                                - 'question': A description of the join operation.
                                - 'sql_tag': A tag representing the type of the SQL operation. In this case 'JOIN-PROJECT-ALL'.

        Note:
            The number of tests generated is equal to the number of foreign keys in the 'table' object.
            If there are no foreign keys, it returns an empty list.
        """

        table_name = table.tbl_name
        tests = []

        # Loop through foreign keys to create JOIN queries
        for foreign_key in table.foreign_keys:
            table_name_2 = foreign_key['child_table'].tbl_name
            parent_col = foreign_key['parent_column']
            child_col = foreign_key['child_column']
            
            # Select up to 5 columns for ORDER BY
            if len(rem_columns) > 0:
                columns = rem_columns

            columns = utils_list_sample(columns, k=10, val=self.column_to_include)
            operations = [('ASC', 'ascending'), ('DESC', 'descending')]
            limit = 5
            
            for col in columns:
                # Check if column exists in parent table
                col_exists_in_parent = col in table.tbl_col2metadata
                col_exists_in_child = col in foreign_key['child_table'].tbl_col2metadata
                
                # Determine which table alias to use for the column
                if col_exists_in_parent and not col_exists_in_child:
                    table_alias = "T1"
                elif col_exists_in_child and not col_exists_in_parent:
                    table_alias = "T2"
                elif col_exists_in_parent and col_exists_in_child:
                    # If column exists in both tables, use parent table (T1) as default
                    table_alias = "T1"
                else:
                    # Skip this column if it doesn't exist in either table
                    continue
                
                for operation in operations:
                    # SQL query with JOIN, ORDER BY, and LIMIT - using table alias to avoid ambiguity
                    query_str = (
                        f"SELECT * FROM `{table_name}` AS T1 "
                        f"JOIN `{table_name_2}` AS T2 ON T1.`{parent_col}` = T2.`{child_col}` "
                        f"ORDER BY {table_alias}.`{col}` {operation[0]} LIMIT {limit}"
                    )
                    
                    # Natural language description
                    question_str = (
                        f"Join all records from table `{table_name}` with table `{table_name_2}` on column `{parent_col}`, "
                        f"then list the top {limit} rows sorted by `{col}` in {operation[1]} order."
                    )
                    
                    test = SingleQA(
                        query=query_str,
                        question=question_str,
                        sql_tag='JOIN-ORDERBY-LIMIT'
                    )
                    tests.append(test)

        return tests

    def generate_join_project_single(self, table: ConnectorTable, columns, rem_columns) -> list[SingleQA]:
        """
            Generates a list of `SingleQA` objects based on JOIN queries derived from the given `table`.
            The JOIN queries focus on categorical columns from the parent and child tables that are connected through a foreign key.
            Each unique pair of (parent column, child column), excluding the foreign key pair itself, is used to create a JOIN query.
            Therefore, the total number of generated tests is approximately:
            len(foreign_keys) * (len(cat_cols_parent) - 1) * (len(cat_cols_child) - 1).

            Args:
                table (ConnectorTable): The table for which to generate the JOIN queries and corresponding questions.

            Returns:
                list[SingleQA]: A list of `SingleQA` objects, each containing:
                                - the SQL JOIN query,
                                - a natural language description of the query, and
                                - a corresponding SQL tag.

            Notes:
                - If the number of categorical columns exceeds 3 in either the parent or child table, only a sample of 3 columns
                is selected to limit the number of generated tests.
                - If a categorical column matches the foreign key column in either the parent or child table, that column pair is skipped.
        """
        table_name = table.tbl_name
        tests = []

        # Sample up to 3 categorical columns from the parent table
        cat_cols_parent = list(table.cat_col2metadata.keys())
        cat_cols_parent = utils_list_sample(cat_cols_parent, k=10, val=self.column_to_include)

        # Loop through all foreign keys to create JOIN queries
        for foreign_key in table.foreign_keys:
            table_name_2 = foreign_key['child_table'].tbl_name
            parent_col = foreign_key['parent_column']
            child_col = foreign_key['child_column']

            # Sample up to 3 categorical columns from the child table
            cat_cols_child = list(foreign_key['child_table'].cat_col2metadata.keys())
            cat_cols_child = utils_list_sample(cat_cols_child, k=10, val=self.column_to_include)

            limit = 5  # Limit number of rows
            operations = [('ASC', 'ascending'), ('DESC', 'descending')]

            # Combine all sampled columns from parent and child tables
            if len(rem_columns) > 0:
                columns = rem_columns

            columns = utils_list_sample(columns, k=5, val=self.column_to_include)

            for col in columns:
                # Check if column exists in parent table or child table
                col_exists_in_parent = col in table.tbl_col2metadata
                col_exists_in_child = col in foreign_key['child_table'].tbl_col2metadata
                
                # Determine which table alias to use for the column
                if col_exists_in_parent and not col_exists_in_child:
                    table_alias = "T1"
                elif col_exists_in_child and not col_exists_in_parent:
                    table_alias = "T2"
                elif col_exists_in_parent and col_exists_in_child:
                    # If column exists in both tables, use parent table (T1) as default
                    table_alias = "T1"
                else:
                    # Skip this column if it doesn't exist in either table
                    continue
                
                for op, op_name in operations:
                    # SQL query: select a single column with JOIN, ORDER BY, LIMIT - using table alias
                    query_str = (
                        f'SELECT {table_alias}.`{col}` FROM `{table_name}` AS T1 '
                        f'JOIN `{table_name_2}` AS T2 ON T1.`{parent_col}` = T2.`{child_col}` '
                        f'ORDER BY {table_alias}.`{col}` {op} '
                        f'LIMIT {limit}'
                    )

                    # Natural language description
                    question_str = (
                        f'Join `{table_name}` and `{table_name_2}` on `{parent_col}`, '
                        f'select only the column `{col}`, '
                        f'sort it in {op_name} order, and return the top {limit} rows.'
                    )

                    test = SingleQA(
                        query=query_str,
                        question=question_str,
                        sql_tag='JOIN-ORDERBY-SINGLE-COL'
                    )
                    tests.append(test)

        return tests

    def generate_advanced_join_order(self, columns, tbl_name, rem_columns, table: ConnectorTable) -> list[SingleQA]:
        from itertools import combinations
        import random

        tests = []
        operations = [('ASC', 'ascending'), ('DESC', 'descending')]

        # Filter columns if rem_columns is provided
        if len(rem_columns) > 0:
            columns = [c for c in columns if c in rem_columns]

        # Generate all combinations of 2 to 3 columns for ordering
        all_combinations = []
        k = 10  # max size of combinations
        n = 10  # max number of combinations to sample
        for r in range(2, k + 1):
            all_combinations.extend(combinations(columns, r))
        random.shuffle(all_combinations)
        n = min(n, len(all_combinations))
        col_combinations = all_combinations[:n]

        limit = 5  # limit rows

        # Loop through all foreign keys to include JOIN logic
        for foreign_key in table.foreign_keys:
            table_name_2 = foreign_key['child_table'].tbl_name
            parent_col = foreign_key['parent_column']
            child_col = foreign_key['child_column']

            for col_combination in col_combinations:
                # Build ORDER BY clause with proper table aliases
                order_by_parts = []
                for col in col_combination:
                    # Check if column exists in parent table or child table
                    col_exists_in_parent = col in table.tbl_col2metadata
                    col_exists_in_child = col in foreign_key['child_table'].tbl_col2metadata
                    
                    # Determine which table alias to use for the column
                    if col_exists_in_parent and not col_exists_in_child:
                        table_alias = "T1"
                    elif col_exists_in_child and not col_exists_in_parent:
                        table_alias = "T2"
                    elif col_exists_in_parent and col_exists_in_child:
                        # If column exists in both tables, use parent table (T1) as default
                        table_alias = "T1"
                    else:
                        # Skip this column if it doesn't exist in either table
                        continue
                    
                    order_by_parts.append(f'{table_alias}.`{col}`')
                
                # Skip if no valid columns found
                if not order_by_parts:
                    continue
                
                for op, op_name in operations:
                    # Build the SQL query: JOIN + ORDER BY combination of columns + LIMIT
                    query_str = (
                        f'SELECT * FROM `{tbl_name}` AS T1 '
                        f'JOIN `{table_name_2}` AS T2 ON T1.`{parent_col}` = T2.`{child_col}` '
                        f'ORDER BY {", ".join(order_by_parts)} {op} LIMIT {limit}'
                    )

                    # Natural language description for LLM
                    question_str = (
                        f'Join `{tbl_name}` with `{table_name_2}` on `{parent_col}`. '
                        f'Sort all rows by the columns {", ".join(col_combination)} in {op_name} order, '
                        f'and return the first {limit} rows.'
                    )

                    tests.append(
                        SingleQA(
                            query=query_str,
                            question=question_str,
                            sql_tag='JOIN-ORDERBY-ADVANCED'
                        )
                    )

        return tests