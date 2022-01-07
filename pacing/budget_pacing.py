import random
from model.model import Model
from pmdarima.arima import auto_arima

class BudgetPacing(Model):
    def __init__(self, train, params, budget, ts_begin = 0, ptr = 0.005):
        self.params = params
        self.train = train
        self.ptr = ptr
        self.sloct_budget = []
        self.history = []
        self.budget = budget
        self.ts_cur = ts_begin
        self.cycle = 96 # = 4 * 24 

    def fit(self):
        self.model = auto_arima(
            self.train,
            seasonal=self.params["seasonal"],
            start_p=self.params["start_p"],
            start_q=self.params["start_q"],
            max_p=self.params["max_p"],
            max_q=self.params["max_q"],
            stepwise=True,
        )
        self.history = [x for x in self.train]
        self.model.fit(self.history)

    def predict_budget(self, test, cnt):
        self.history.append(test[cnt])
        self.model.fit(self.history)
        preds = self.model.predict(n_periods=self.cycle-cnt)
        self.slot_budget = [x/sum(preds) * self.budget for x in preds]
        return self.slot_budget
    
    def predict(self, n_periods):
        return self.model.predict(n_periods)

    def is_need_update(self, ts, slot_cost):
        if ts == self.ts_cur or slot_cost == 0:
            return False
        else:
            self.ts_cur = ts
            return True

    def update_ptr(self, slot_budget, slot_cost, upper_factor, lower_factor):
        if slot_cost < slot_budget*upper_factor:
            self.ptr *= (1/upper_factor) if self.ptr < 1 else self.ptr 
        if slot_cost > slot_budget*lower_factor:
            self.ptr *= (1/lower_factor)        

    def update(self, budget, slot_cost, slot_budget, test, cnt):
        if self.cycle - cnt > 4:
            self.update_ptr(slot_budget, slot_cost, 0.9, 1.1)
        else:
            self.update_ptr(slot_budget, slot_cost, 0.8, 1.2)
        slot_budget = self.predict_budget(test, cnt)
        self.budget = budget
        return slot_budget
    
    def is_try_bidding(self):
        return random.random() < self.ptr
    
    def get_ptr(self):
        return self.ptr