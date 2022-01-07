from model.model import Model
class PID(Model):
    def __init__(self, ref, first_round, sec_round, phi):
        self.ref = ref
        self.first_round = first_round
        self.sec_round = sec_round
        self.phi = phi
        self.para_p = 0.0005
        self.para_i = 0.000001
        self.para_d = 0.0001
        self.error_sum = 0
        self.min_phi = -2
        self.max_phi = 5
    
    def update(self, round, ecpcs):
        if self.first_round and (not self.sec_round):
            self.phi = 0.0
            self.first_round = False
            self.sec_round = True
        elif self.sec_round and (not self.first_round):
            error = self.ref - ecpcs[round-1]
            self.error_sum += error
            self.phi = self.para_p*error + self.para_i*self.error_sum
            self.sec_round = False
        else:
            error = self.ref - ecpcs[round-1]
            self.error_sum += error
            self.phi = self.para_p*error + self.para_i*self.error_sum + self.para_d*(ecpcs[round-2]-ecpcs[round-1])    
    
    def check_bound(self):
        if self.phi <= self.min_phi:
            self.phi = self.min_phi
        elif self.phi >= self.max_phi:
            self.phi = self.max_phi
        return self.phi