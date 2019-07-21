"""Stub unit tests."""


def hello_world():
    """Hello World."""
    return "Hello World"


class TestHelloWorld:
    def test_return_type(self):
        """Test return type of hello_world()."""
        output = hello_world()
        assert isinstance(output, str)

    def test_return_value(self):
        """Test return value of hello_world()."""
        output = hello_world()
        assert output == "Hello World"
