#!/usr/bin/env python3

""" Task: write a class which receives
    - the field size;
    - the coordinates of all already purchased squares;
    - the coordinates and cost of all squares with cost different than zero;
    - the coordinataes and #roses of all squares with at least a single rose;
    - the desired garden shape (H,W).

    For any given garden shape and budget, the class should be able to calculate coordinates(r,c) such that the
    axis-aligned rectangle of shape H*W with its top-left corner located at (r,c),
    - is fully containede in the field,
    - costs at most the budget,
    - do not contain any already purchased squares, and,
    - the number of roses is maximal.

    The class should implement IRoseField interface.
"""
import logging
from  src.IRosesField import IRosesField

logger=logging.getLogger(__name__)

class RosesField(IRosesField):
    """
    This class implements the IRosesField interface (__init__, set_garden_shape,find_best_garden) and several methods
    like as fields_info2matrix, find_squares, total_report.

    The flow behavior:
           - for given rose location and cost dictionaries the "matrix"-view  (list of lists) are created. It is
           algorithmically more  convenient to shift (right and down) the "garden"-rectangle over "field"-rectangle than
           to use a sparse dictionary. The input data for task is more convenient to define as sparse dictionaries. The
           "dict to matrix" transformation is carried out on init stage and do not influence to performence.

           - the complexity of garden search O(.) in the worst case no more
           (field_width -garden_width+1) * (field_height -garden_heaigth+1)  or O(n)<n^2.

           - the gardens with cost more than a budget are ignored.

           - from several identical solutions one is selected closest to "nordwestern" corner.

           - all generated sequences of squres with amount roses ana costs are logged in "main.log"

           - only "nordwestern" corner (row,column) for garden sequence si put on terminal.


    """

    def __init__(self, field_width: int, field_height: int, purchased_squares: list,
                 location_roses: dict, location_costs: dict,
                 garden_width: int, garden_height: int):

        """Constructor"""
        self.field_width = field_width
        self.field_height = field_height
        self.purchased_squares = purchased_squares
        self.location_roses = location_roses
        self.location_costs = location_costs
        self.garden_width = garden_width
        self.garden_height = garden_height
        self.selected_squares = []
        self._log = logger

        self.matrix_location = None
        self.matrix_costs = None
        self.fields_info2matrix()


    def set_garden_shape(self, garden_width: int, garden_height: int):
        """Setting the desired garden’s shape"""
        self.garden_width = garden_width
        self.garden_height = garden_height
        self.selected_squares = []    # reset previously selected squares


    def find_best_garden(self, budget: float) -> tuple:
        """Finding the best garden given budget"""
        self.find_squares(budget)
        (r,c),amount,total_cost = self.total_report()
        self.selected_squares = []
        return (r,c)

    def __str__(self):
        smsg =f""" 
        
Fields: {self.field_height } x {self.field_width}  
Garden: {self.garden_height} x {self.garden_width}
Roses location   : {self.location_roses}
Cost             : {self.location_costs}
Purchased squares: {self.purchased_squares}


              """
        return smsg

    def fields_info2matrix(self):
        self.matrix_location=[[0 for j in range(self.field_width) ]  for i in range(self.field_height)]
        self.matrix_costs = [[0 for j in range(self.field_width) ]  for i in range(self.field_height)]

        for key,value in self.location_roses.items():
            (r,c)=key
            self.matrix_location[r][c]=value

        for key,value in self.location_costs.items():
            (r,c)=key
            self.matrix_costs[r][c]=value

        for item in self.purchased_squares:
            (r,c)=item
            self.matrix_costs[r][c]=-1.0 * self.matrix_costs[r][c]
            self.matrix_location[r][c] = -1 * self.matrix_location[r][c]
        self._log.info("Fields :\n{}".format(self.matrix_location))
        self._log.info("Costs per field:\n{}".format(self.matrix_costs))


    def find_squares(self,budget:float):
        pass

        r = 0
        c = 0

        while r<self.field_height and c <self.field_width:
            dutyrectangle = False
            expensiverectangle = False
            # is garden_rectangle ouf of fields rectangle?
            if c+self.garden_width>self.field_width or r + self.garden_height>self.field_height:
                break

            if r>0 and r+self.garden_height>self.field_height:  # can not shift down
                c=c+1
                r=0
                if c+self.garden_width>self.field_width: # can not shif right
                    break
            cost=0.0
            amount =0
            d={}

            for i in range(r,r+self.garden_height):
                for j in range(c,c+self.garden_width):
                    if self.matrix_location[i][j] <0 or self.matrix_costs[i][j]<0.0:
                        dutyrectangle = True
                        break
                    cost=cost + self.matrix_costs[i][j]
                    amount = amount + self.matrix_location[i][j]

                    if cost > budget:
                        expensiverectangle = True

                if dutyrectangle:
                    break
            # the garden with expensive cost should be ignored.
            if expensiverectangle:
                self._log.info("Expensive cost: {} for rectangle wich started at ({},{})".format(cost,r,c))

            if not dutyrectangle and not expensiverectangle:
                # save
                d[(r,c)]=(cost,amount)
                self.selected_squares.append(d)
            c=c+1
            if c+self.garden_width>self.field_width: # shift right on one column and start with 0-row
                                                     #shif down on one row and start with 0-column
                r=r+1
                c=0


        return

    def total_report(self)->(tuple,int,float):
        pass
        if not self.selected_squares:
            self._log.info("O-o-ops! We could not find a suitable squares with roses")
            return (-1,-1),-1,0.0
        number_of_roses=0
        top_left=(-1,-1)
        total_cost = 0.0
        for item in self.selected_squares:
            (key,val), =item.items()
            (r,c)=key
            (cost,amount)=val
            if amount>number_of_roses:
                number_of_roses=amount
                total_cost=cost
                top_left=key
        self._log.info("Great square {}-({},{}) squares with {} roses for only {} N.I.S".format(key,
                                    self.garden_height,self.garden_width, amount, total_cost))
        return top_left,amount,total_cost





