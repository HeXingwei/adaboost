__author__ = 'hxw'
import numpy as np
class adaboost():
	def __init__(self,traindata,epo=100,numstep=100):
		self.x=traindata[:,:-1]
		self.y=traindata[:,-1]
		self.epo=epo
		self.numstep=numstep
		self.classfier=[]
		self.m,self.n=self.x.shape
		self.D=np.ones((self.m))/self.m
	def tree_classify(self,x,dimen_i,thresh_val,thresh_op):
		predict=np.ones((x.shape[0],1))
		if thresh_op=="lt":
			predict[x[:,dimen_i]<=thresh_val]=-1
		else:
			predict[x[:,dimen_i]>thresh_val]=-1
		return predict
	def build_tree(self):
		min_error=np.inf
		best_tree={}
		best_predict=None
		for i in range(self.n):
			min_val=self.x[:,i].min()
			max_val=self.x[:,i].max()
			step_val=(max_val-min_val)/self.numstep
			for j in range(-1,self.numstep+1):
				thresh_val=min_val+j*step_val
				for op in ["lt","gt"]:
					predict_val=self.tree_classify(self.x,i,thresh_val,op)
					temp=np.zeros(self.m)
					temp[predict_val[:,0]!=self.y]=1
					error=temp.dot(self.D)
					if min_error>error:
						min_error=error
						best_tree["op"]=op
						best_tree["dim"]=i
						best_tree["thresh_val"]=thresh_val
						best_predict=predict_val.copy()
		return min_error,best_tree,best_predict

	def fit(self):
		agg_predict_y=np.zeros((self.m,1))
		for i in range(self.epo):
			error,best_tree,predict_y=self.build_tree()
			alpha=0.5*np.log((1-error)/max(error,1e-16))
			best_tree["alpha"]=alpha
			self.classfier.append(best_tree)
			#update self.D
			self.D=self.D*np.exp(-alpha*self.y*predict_y[:,0])
			self.D/=self.D.sum()
			#calc training error of all classifiers, if this is 0 quit for loop early (use break)
			agg_predict_y+=alpha*predict_y
			total_error=(np.sign(agg_predict_y[:,0])!=self.y).sum()/float(self.m)
			print "total error: %f"%total_error
			if total_error==0:
				break
	def predict(self,x,y):
		m,n=x.shape
		agg_predict_y=np.zeros((m,1))
		for classifier in self.classfier:
			predict_y=self.tree_classify(x,classifier["dim"],classifier["thresh_val"],classifier["op"])
			agg_predict_y+=classifier["alpha"]*predict_y
		return np.sign(agg_predict_y)
	def accuracy(self,x,y):
		predict=self.predict(x,y)
		m=y.shape[0]
		accuracy=(predict[:,0]==y).sum()/float(m)
		print "test accuracy is %f"%accuracy

def loaddata(path):
	data=np.loadtxt(path)
	return data
train_path="D:\\SelfLearning\\Machine Learning\\MachineLearningInAction\\machinelearninginaction\\Ch07\\horseColicTraining2.txt"
test_path="D:\\SelfLearning\\Machine Learning\\MachineLearningInAction\\machinelearninginaction\\Ch07\\horseColicTest2.txt"
train_data=loaddata(train_path)
test_data=loaddata(test_path)
ada=adaboost(train_data)
ada.fit()
ada.accuracy(test_data[:,:-1],test_data[:,-1])
