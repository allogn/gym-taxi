class Driver:

    def __init__(self, driver_id, income_bound = None):
        self.income = 0
        self.position = -1
        self.driver_id = driver_id
        self.status = 1 # 1 : online (idle), 0 : traveling
        self.income_bound = income_bound
        self.not_idle_periods = 0

    def inc_not_idle(self):
        self.not_idle_periods += 1
        return self.get_not_idle_periods()

    def get_not_idle_periods(self):
        return self.not_idle_periods

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
