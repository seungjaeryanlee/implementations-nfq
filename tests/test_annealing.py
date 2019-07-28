"""Test utils.get_linear_anneal_func."""
import types

import pytest

from utils import get_linear_anneal_func


class TestGetLinearAnnealFunc:
    @pytest.mark.parametrize("start, end, steps", [(1, 0, 100), (2, 0, 10)])
    def test_return_type(self, start, end, steps):
        """Test the return type of get_linear_anneal_func."""
        linear_anneal_func = get_linear_anneal_func(start, end, steps)
        assert isinstance(linear_anneal_func, types.FunctionType)

    @pytest.mark.parametrize("start, end, steps", [(1, 0, 100), (2, 0, 10)])
    def test_return_func_return_type(self, start, end, steps):
        """Test the return type of linear_anneal_func."""
        linear_anneal_func = get_linear_anneal_func(start, end, steps)
        value = linear_anneal_func(end)
        assert isinstance(value, float)

    @pytest.mark.parametrize("start, end, steps", [(1, 0, 100), (2, 0, 10)])
    def test_return_func_start_value(self, start, end, steps):
        """Test the start value of linear_anneal_func."""
        linear_anneal_func = get_linear_anneal_func(start, end, steps)
        value = linear_anneal_func(0)
        assert value == start

    @pytest.mark.parametrize("start, end, steps", [(1, 0, 100), (2, 0, 10)])
    def test_return_func_mid_value(self, start, end, steps):
        """Test the middle value of linear_anneal_func."""
        linear_anneal_func = get_linear_anneal_func(start, end, steps)
        value = linear_anneal_func(steps / 2)
        assert value == (start - end) / 2

    @pytest.mark.parametrize("start, end, steps", [(1, 0, 100), (2, 0, 10)])
    def test_return_func_end_value(self, start, end, steps):
        """Test the end value of linear_anneal_func."""
        linear_anneal_func = get_linear_anneal_func(start, end, steps)
        value = linear_anneal_func(steps)
        assert value == end

    @pytest.mark.parametrize("start, end, steps", [(1, 0, 100), (2, 0, 10)])
    def test_return_func_after_end_value(self, start, end, steps):
        """Test the value of linear_anneal_func after end_steps."""
        linear_anneal_func = get_linear_anneal_func(start, end, steps)
        value = linear_anneal_func(steps + 1)
        assert value == end
