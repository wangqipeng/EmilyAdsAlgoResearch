

class RBbidding:
	up_precision = 1e-10
	zero_precision = 1e-12
	def __init__(self, camp_info, opt_obj, gamma):
		self.cpm = camp_info["cost_train"] / camp_info["imp_train"]
		self.theta_avg = camp_info["clk_train"] / camp_info["imp_train"]
		self.opt_obj = opt_obj
		self.gamma = gamma
		self.v1 = self.opt_obj.v1
		self.v0 = self.opt_obj.v0
		self.V = []
		self.D = []

    def optimal_value_function(self, mp_pdf, avg_theta, N, B):
        V = [0] * (B + 1)
        nV = [0] * (B + 1)
		V_max = 0
		V_inc = 0

		a_max = max_bid
		for b in range(0, a_max + 1):
			V_inc += m_pdf[b] * avg_theta 
		for n in range(1, N):
			a = [0] * (B + 1)
			bb = B - 1
			for b in range(B, 0, -1):
				while bb >= 0 and (avg_theta + (V[bb] - V[b])) >= 0:# g(delta)
					bb -= 1
				if bb < 0:
					a[b] = min(max_bid, b)
				else:
					a[b] = min(max_bid, b - bb - 1)

			for b in range(0, B):
				V_out.write("{}\t".format(V[b]))
			V_out.write("{}\n".format(V[B]))

			V_max += V_inc
			for b in range(1, B + 1):
				nV[b] = V[b]
				for delta in range(0, a[b] + 1):
					nV[b] += m_pdf[delta] * (avg_theta + (V[b - delta] - V[b]))  #formula:（8）
				if abs(nV[b] - V_max) < self.up_precision:
					for bb in range(b + 1, B + 1):
						nV[bb] = V_max
					break
			V = nV[:]
		for b in range(0, B):
			V_out.write("{0}\t".format(V[b]))
		V_out.write("{0}\n".format(V[B]))
		V_out.flush()
		V_out.close()

	def bid(self, n, b, theta, max_bid):
		a = 0
		for delta in range(1, min(b, max_bid) + 1):
			if theta + (self.V[n - 1][b - delta] - self.V[n - 1][b])  >= 0: 
				a = delta
			else:
			    break

		return a