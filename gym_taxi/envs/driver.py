class Driver:

    def __init__(self, driver_id, income_bound = None):
        self.income = 0
        self.position = -1
        self.driver_id = driver_id
        self.status = 1 # 1 : online (idle), 0 : traveling
        self.income_bound = income_bound
        self.idle_period = 0 # inversed!!!! since we need to maximize this, then idle is how much he was not idle!

    def inc_idle(self):
        self.idle_period += 1
        return self.get_idle_period()

    def get_idle_period(self):
        return self.idle_period

    def add_income(self, s):
        t = self.get_income()
        self.income += s
        return self.get_income() - t

    def get_income(self):
        if self.income_bound is None:
            return self.income
        else:
            return min(self.income_bound, self.income)

    def __str__(self):
        return "Driver {}: Income {}, Position {}, Status {}, Bound {}".format(self.driver_id, self.income, self.position, self.status, self.income_bound)
