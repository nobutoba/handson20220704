from typing import List


def doctest_sample(table_data: List) -> None:
    """Print first element in given table.

    Args:
        table_data (List): A list of table items.

    >>> doctest_sample(['apple', 'banana'])
    apple
    """

    assert table_data
    print(table_data[0])


if __name__ == "__main__":
    import unittest

    unittest.main()
