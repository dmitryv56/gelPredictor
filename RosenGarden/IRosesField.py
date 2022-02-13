#!/usr/bin/env python3

""" Roses Garden Interface, where
    - "field_width" and "field_height" are the dimensions of the field.
    - "purchased_squares" - the locations of the squares already purchased (list of (r,c) tuples).
    - "location_roses" - a dict (key -location(r,c), value -#roses).
    - "location_costs" - a dict(key - location(r,c), value - cost).
    - "garden_width" and "garden_height" are the dimensions of the desired garden.
    - "budget" is maximum allowed cost of garden.
    - the return value of find_best_garden should have r(row) and c(colunb) coordinates of the solution, in this order.
    - "r" must be less than "field_height", "c" must be less than "field_width".
    def __init__(self, field_width: int, field_height: int, purchased_squares: list[tuple[int, int]],
                 location_roses: dict[tuple[int, int], int], location_costs: dict[tuple[int, int], float],
                 garden_width: int, garden_height: int):
"""

class IRosesField:

    def __init__(self, field_width: int, field_height: int, purchased_squares: list,
                 location_roses: dict, location_costs: dict,
                 garden_width: int, garden_height: int):

        """Constructor"""
        pass

    def set_garden_shape(self, garden_width: int, garden_height: int):
        """Setting the desired garden’s shape"""
        pass

    def find_best_garden(self, budget: float) -> tuple:
        """Finding the best garden given budget"""
        pass