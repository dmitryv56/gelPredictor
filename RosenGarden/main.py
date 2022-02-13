#!/usr/bin/env python3

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rosesfield import RosesField

MAX_LOG_SIZE_BYTES = 1024 * 1024
BACKUP_COUNT = 2
MAIN_LOG = Path(Path(__file__).stem ).with_suffix(".log")

size_handler=RotatingFileHandler(MAIN_LOG, mode='a', maxBytes=MAX_LOG_SIZE_BYTES, backupCount=BACKUP_COUNT )

logger=logging.getLogger()

logger.setLevel(logging.DEBUG)
log_formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  '%m/%d/%Y %I:%M:%S %p')
size_handler.setFormatter(log_formatter)
logger.addHandler(size_handler)

def init_task()->(int,int, list, dict, dict):
    field_width = 3
    field_height = 3
    purchased_squares = []
    purchased_squares.append((0, 2))

    location_roses = {}
    location_roses[(0, 0)] = 1
    location_roses[(0, 1)] = 0
    location_roses[(0, 2)] = 10
    location_roses[(2, 0)] = 1
    location_roses[(2, 2)] = 5

    location_costs = {}
    location_costs[(0, 1)] = 2.0
    location_costs[(0, 2)] = 1.0
    location_costs[(2, 0)] = 2.0
    location_costs[(2, 2)] = 3.0
    pass
    garden_width  = 3
    garden_height = 1

    return field_width, field_height, garden_width, garden_height, purchased_squares, location_roses, location_costs

def example_task()->(int,int, list, dict, dict):
    field_width = 4
    field_height = 6
    purchased_squares = []
    purchased_squares.append((0, 2))
    purchased_squares.append((3, 0))
    purchased_squares.append((3, 1))
    purchased_squares.append((3, 2))

    location_roses = {}
    location_roses[(0, 0)] = 1     # row 0
    location_roses[(0, 2)] = 10
    location_roses[(1, 0)] = 4     # row 1
    location_roses[(1, 3)] = 2
    location_roses[(2, 1)] = 4     # row 2
    location_roses[(2, 2)] = 5
    location_roses[(2, 3)] = 1
    location_roses[(3, 0)] = 2    # row 3
    location_roses[(3, 2)] = 9
    location_roses[(4, 0)] = 3    # row 4
    location_roses[(4, 2)] = 14
    location_roses[(5, 0)] = 5    # row 5
    location_roses[(5, 2)] = 1
    location_roses[(5, 3)] = 100


    location_costs = {}
    location_costs[(0, 0)] = 0  # row 0
    location_costs[(0, 2)] = 1
    location_costs[(1, 0)] = 1  # row 1
    location_costs[(1, 3)] = 5
    location_costs[(2, 1)] = 2  # row 2
    location_costs[(2, 2)] = 3
    location_costs[(2, 3)] = 1
    location_costs[(3, 0)] = 1  # row 3
    location_costs[(3, 2)] = 2
    location_costs[(4, 0)] = 2  # row 4
    location_costs[(4, 2)] = 20
    location_costs[(5, 0)] = 1  # row 5
    location_costs[(5, 2)] = 10
    location_costs[(5, 3)] = 1
    pass
    garden_width  = 3
    garden_height = 1

    return field_width, field_height, garden_width, garden_height, purchased_squares, location_roses, location_costs

def main():
    field_width, field_height, garden_width, garden_height, purchased_squares, location_roses, location_costs = \
        example_task()

    roses_field = RosesField(field_width, field_height, purchased_squares, location_roses, location_costs, garden_width,
                             garden_height)
    print(roses_field.__str__())
    logger.info(roses_field.__str__())

    print(roses_field.find_best_garden(1.0))
    print(roses_field.find_best_garden(2.0))
    print(roses_field.find_best_garden(5.0))
    print(roses_field.find_best_garden(20.0))

    roses_field.set_garden_shape(2, 2)
    print(roses_field.__str__())
    logger.info(roses_field.__str__())

    print(roses_field.find_best_garden(1.0))
    print(roses_field.find_best_garden(2.0))
    print(roses_field.find_best_garden(5.0))
    print(roses_field.find_best_garden(20.0))

    return

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logger.info("Start...")
    main()
    logger.info("... finish.")


