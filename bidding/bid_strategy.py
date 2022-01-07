import math

bid_upper=301

# λ: λ controls the general scale if bidding price: when  λ is higher, the bid price gets lower  
#c: direct learn c to best fit the winning rate data for each campaign
def estimate_c(bid_wr):
    # model win = b / (b + l)
    ls = range(1, bid_upper)
    min_loss = 9E50
    optimal_c = -1
    for l in ls:
        loss = 0
        for (bid, win) in bid_wr:
            y = win
            yp = bid * 1.0 / (bid + l)
            loss += (y - yp) * (y - yp)
        if loss < min_loss:
            min_loss = loss
            optimal_c = l
    return optimal_c

class BidStrategy:
    def __init__(self, camp_v, lamb):
        self.lamb = lamb
        self.phi = 1.0 / (1.0 + self.lamb)
        self.camp_v = camp_v

    def set_camp_value(self, v):
        self.camp_v = v

    def set_lamb(self, lamb):
        self.lamb = lamb
        self.gamma = 1.0 / (1.0 + self.lamb)

	# b = 1.0 / (1.0 + mu) * ctr
    def bidding_opt(self, ctr):
        bid_price = int(1.0 / (1.0 + self.lamb) * self.camp_v * ctr * 1E3)
        return bid_price
    
    def bidding_ortb(self, pctr, base_ctr, c, l):
        lamb = base_ctr/l
        return int(math.sqrt(pctr * c / lamb + c * c) - c)

    def bidding_mcpc(self, ecpc, pctr):
        return int(ecpc * pctr)

    def bidding_ecpm(self, base_price, pctr, pcvr):
        return int(base_price * pctr * pcvr)

    @staticmethod
    def bidding_lin(pctr, base_ctr, base_bid):
        return int(pctr * base_bid / base_ctr)
