import numpy as np

class Node:
    def __init__(self, node_id):
        self.node_id = node_id
        self.clear_orders()
        self.clear_drivers()

    def add_orders(self, order_list):
        self.orders += [r for r in order_list]

    def add_driver(self, driver):
        self.drivers.append(driver)

    def clear_orders(self):
        self.orders = []

    def clear_drivers(self):
        self.drivers = []

    def get_driver_num(self):
        return len(self.drivers)

    def get_order_num(self):
        return len(self.orders)

    def select_and_remove_orders(self, number_of_orders):
        assert number_of_orders <= len(self.orders)
        assert number_of_orders >= 0
        if number_of_orders == 0:
            return []
        np.random.shuffle(self.orders)
        selected = self.orders[:number_of_orders]
        self.orders = self.orders[number_of_orders:]
        return selected

    def __str__(self):
        return "Node {}: Drivers {}, Orders {}".format(self.node_id, self.get_driver_num(), self.get_order_num())
