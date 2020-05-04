import copy

class Driver:

    def __init__(self, driver_id, env, income_bound = None):
        self.env = env
        self.DEBUG = env.DEBUG
        self.income = 0
        self.position = -1
        self.driver_id = driver_id
        self.status = 1 # 1 : online (idle), 0 : traveling
        self.income_bound = income_bound
        self.not_idle_periods = 0
        self.history = []

    def inc_not_idle(self):
        self.not_idle_periods += 1
        if self.DEBUG:
            self.history.append("t={}: Non-idle period increased by 1".format(self.env.time))
        return self.get_not_idle_periods()

    def get_not_idle_periods(self):
        return self.not_idle_periods

    def update_position(self, new_position):
        if self.DEBUG:
            self.history.append("t={}: Moving from position {} to {}".format(self.position, new_position, self.env.time))
        self.position = new_position

    def set_inactive(self):
        if self.DEBUG:
            self.history.append("t={}: Status updated to 0".format(self.env.time))
        self.status = 0

    def set_active(self):
        if self.DEBUG:
            self.history.append("t={}: Status updated to 1".format(self.env.time))
        self.status = 1

    def add_income(self, s):
        t = self.get_income()
        self.income += s
        if self.DEBUG:
            self.history.append("t={}: Income increased by {}".format(self.env.time, s))
        return self.get_income() - t

    def get_income(self):
        if self.income_bound is None:
            return self.income
        else:
            return min(self.income_bound, self.income)

    def sync(self, another_driver):
        assert self.driver_id == another_driver.driver_id
        self.history = copy.deepcopy(another_driver.history)
        self.income = another_driver.income
        self.income_bound = another_driver.income_bound
        self.position = another_driver.position
        self.status = another_driver.status
        self.not_idle_periods = another_driver.not_idle_periods

    def __str__(self):
        return "t={}: Driver {}: Income {}, Position {}, Status {}, IncBound {}.\n History: {}".format(
            self.env.time, self.driver_id, self.income, self.position, self.status, self.income_bound, self.history
            )
