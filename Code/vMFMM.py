from pickle import load
import numpy as np
import scipy
if tuple(map(int, scipy.__version__.split('.'))) < (1, 0, 0):
    from scipy.misc import logsumexp
else:
    from scipy.special import logsumexp
import time

#? Should try fuzzy/xmeans/kmeans with PCA/meanshift algorithm - should dbscan or density based clustering better for noisy data?

def normalize_features(features):
	'''features: n by d matrix'''
	assert(len(features.shape)==2)
	norma=np.sqrt(np.sum(features ** 2, axis=1).reshape(-1, 1))+1e-6
	return features/norma

class vMFMM:
	def __init__(self, cls_num, init_method = 'random', load_vc=None):
		self.cls_num = cls_num # 512- vc_num
		self.init_method = init_method
		self.load_vc = load_vc


	def fit(self, features, kappa, max_it=300, tol = 5e-5, normalized=False, verbose=True):
		self.features = features #* [512] element size, total size=24601600
		if not normalized:
			self.features = normalize_features(features)

		self.n, self.d = self.features.shape
		self.kappa = kappa #/ What is this?

		self.pi = np.random.random(self.cls_num)
		self.pi /= np.sum(self.pi) #/ pi nows sums to one - membership of each vc? - mixture proportion?
		if self.init_method =='random':
			self.mu = np.random.random((self.cls_num, self.d))
			self.mu = normalize_features(self.mu)
		elif self.init_method =='k++':
			centers = []
			centers_i = []

			if self.n > 50000:
				rdn_index = np.random.choice(self.n, size=(50000,), replace=False)
			else:
				rdn_index = np.array(range(self.n), dtype=int)  #* (48050, )
			
			#* COSINE DISTANCE
			cos_dis = 1-np.dot(self.features[rdn_index], self.features[rdn_index].T) #* [48050, 48050]

			centers_i.append(np.random.choice(rdn_index)) # chooses a value between 0 and 48050
			centers.append(self.features[centers_i[0]]) #* center initialized with a randomly chosen feature
			for i in range(self.cls_num-1):

				cdisidx = [np.where(rdn_index==cci)[0][0] for cci in centers_i]
				prob = np.min(cos_dis[:,cdisidx], axis=1)**2 #? what's this?
				prob /= np.sum(prob)
				centers_i.append(np.random.choice(rdn_index, p=prob))
				centers.append(self.features[centers_i[-1]])

			self.mu = np.array(centers) #* [512, 512] i.e. 512 1D features of length 512
			del(cos_dis)
		# elif self.init_method =='update':
		# 	#/ Load old VCs and update them with new data
		# 	assert(self.load_vc is not None)
		# 	with open(self.load_vc['mu'], 'rb') as fh:
		# 		self.mu = load(fh)
		# 	with open(self.load_vc['p'], 'rb') as fh:
		# 		self.p = load(fh)

		self.mllk_rec = [] #/ IS THIS MAXIMUM LIKELIHOOD?
		for itt in range(max_it):
			_st = time.time()
			self.e_step()
			self.m_step()
			_et = time.time()

			self.mllk_rec.append(self.mllk)
			if len(self.mllk_rec)>1 and self.mllk - self.mllk_rec[-2] < tol:
				if verbose:print("early stop at iter {0}, llk {1}".format(itt, self.mllk))
				break


	def fit_soft(self, features, p, mu, pi, kappa, max_it=300, tol = 1e-6, normalized=False, verbose=True):
		self.features = features
		if not normalized:
			self.features = normalize_features(features)

		self.p = p
		self.mu = mu
		self.pi = pi
		self.kappa = kappa

		self.n, self.d = self.features.shape

		self.mllk_rec = []
		for itt in range(max_it):
			if verbose and itt%10:print("EM Step:{}\{}".format(itt+1,max_it))
			self.e_step()
			self.m_step()

			self.mllk_rec.append(self.mllk)
			if len(self.mllk_rec)>1 and self.mllk - self.mllk_rec[-2] < tol:
				if verbose:print("early stop at iter {0}, llk {1}".format(itt, self.mllk))
				break

	def fit_map(self, features, pre_mu, pre_pi, kappa, reg=0.3, max_it=300, tol = 5e-5, normalized=False, verbose=True):
		self.features = features #* [512] element size, total size=24601600
		if not normalized:
			self.features = normalize_features(features)

		self.n, self.d = self.features.shape
		self.kappa = kappa #/ What is this?
		self.reg = reg #/ regularisation constant for prior term in map

		self.pi = np.random.random(self.cls_num)
		self.pi /= np.sum(self.pi) #/ pi nows sums to one - membership of each vc? - mixture proportion?
		if self.init_method =='random':
			self.mu = np.random.random((self.cls_num, self.d))
			self.mu = normalize_features(self.mu)
		elif self.init_method =='k++':
			centers = []
			centers_i = []

			if self.n > 50000:
				rdn_index = np.random.choice(self.n, size=(50000,), replace=False)
			else:
				rdn_index = np.array(range(self.n), dtype=int)  #* (48050, )
			
			#* COSINE DISTANCE
			cos_dis = 1-np.dot(self.features[rdn_index], self.features[rdn_index].T) #* [48050, 48050]

			centers_i.append(np.random.choice(rdn_index)) # chooses a value between 0 and 48050
			centers.append(self.features[centers_i[0]]) #* center initialized with a randomly chosen feature
			for i in range(self.cls_num-1):

				cdisidx = [np.where(rdn_index==cci)[0][0] for cci in centers_i]
				prob = np.min(cos_dis[:,cdisidx], axis=1)**2 #? what's this?
				prob /= np.sum(prob)
				centers_i.append(np.random.choice(rdn_index, p=prob))
				centers.append(self.features[centers_i[-1]])

			self.mu = np.array(centers) #* [512, 512] i.e. 512 1D features of length 512
			del(cos_dis)
		#/ Load old VCs and update them with new data

		# self.pre_p = pre_p
		self.pre_mu = pre_mu
		self.pre_pi = pre_pi
		self.mu = pre_mu #! reassigning intialisation

		self.mllk_rec = [] #/ IS THIS MAXIMUM LIKELIHOOD?
		for itt in range(max_it):
			_st = time.time()
			self.e_map_step()
			self.m_step()
			_et = time.time()

			self.mllk_rec.append(self.mllk)
			print("Iter {0}, llk {1}".format(itt+1, self.mllk))
			if len(self.mllk_rec)>1 and self.mllk - self.mllk_rec[-2] < tol:
				if verbose:
					print("early stop at iter {0}, llk {1}".format(itt+1, self.mllk))
				break


	def e_step(self):
		# update p
		logP = np.dot(self.features, self.mu.T)*self.kappa + np.log(self.pi).reshape(1,-1)  # n by k
		logP_norm = logP - logsumexp(logP, axis=1).reshape(-1,1)
		self.p = np.exp(logP_norm) #/ posterior/likelihood?
		self.mllk = np.mean(logsumexp(logP, axis=1)) #/mean likelihood


	def m_step(self):
		# update pi and mu
		self.pi = np.sum(self.p, axis=0)/self.n

		# fast version, requires more memory
		self.mu = np.dot(self.p.T, self.features)/np.sum(self.p, axis=0).reshape(-1,1)

		self.mu = normalize_features(self.mu)

	def e_map_step(self):
		# update p
		pre_logP = np.dot(self.features, self.pre_mu.T)*self.kappa + np.log(self.pre_pi).reshape(1,-1)  # n by k
		pre_logP_norm = pre_logP - logsumexp(pre_logP, axis=1).reshape(-1,1)
		temp_p = np.exp(pre_logP_norm)

		logP = np.dot(self.features, self.mu.T)*self.kappa + np.log(self.pi).reshape(1,-1)  # n by k
		logP_norm = logP - logsumexp(logP, axis=1).reshape(-1,1)
		# self.p = np.exp(logP_norm)
		self.p = np.exp((logP_norm+(pre_logP_norm*self.reg))/(1+self.reg))
		print("difference between posteriors {}".format(np.mean(np.absolute(temp_p-self.p))))
		self.mllk = np.mean(logsumexp(logP, axis=1)) #/mean likelihood



