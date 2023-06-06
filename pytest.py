import pytest
from unittest.mock import MagicMock, patch

def scan_dynamodb_table(table_name):
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    response = table.scan()
    return response['Items']

@pytest.fixture(scope='module')
def test_table():
    # No need to create a test table using mock
    yield 'test_table'

@patch('boto3.resource')
def test_scan_dynamodb_table(mock_resource, test_table):
    # Mock the DynamoDB resource and table
    table_mock = MagicMock()
    mock_resource.return_value.Table.return_value = table_mock

    # Set the mock response for the scan method
    items = [
        {'id': 1, 'name': 'John'},
        {'id': 2, 'name': 'Jane'},
        {'id': 3, 'name': 'Alice'}
    ]
    table_mock.scan.return_value = {'Items': items}

    # Call the function being tested
    result = scan_dynamodb_table(test_table)

    # Perform assertions on the result
    assert len(result) == len(items)
    assert all(item in result for item in items)
