from sklearn.isotonic import IsotonicRegression as IR
import pickle, os
import numpy as np
import matplotlib
matplotlib.use('PS')
from matplotlib import pyplot as plt

''' take in a tab-delimited file where first column is true values (0=neutral, 1=sweep), and second column is un-calibrated probabilities; as of now, only binary classification is supported'''
class SmoothedIsotonicCalibration:

	def __init__(self):
		if not os.path.isdir('calibration/'):
			os.system('mkdir calibration/')
		self.numbins = 10

	def apply_calibration(self,frac1,filename):
		file = open(filename,'r')
		f = file.read()
		file.close()
		outfilename = os.path.split(filename)[1]
		outfilename = outfilename.replace('.txt','')+'_calibrated'

		f = f.strip().splitlines()
		header = f[0]
		f = f[1:]
		values = []
		header = header.strip().split('\t')
		valix = header.index('uncalibrated')
		oldlines = []
		for line in f:
			oldlines.append(line)
			line = line.strip().split('\t')
			values.append(float(line[valix]))

		M = pickle.load(open('calibration/M_frac1='+str(frac1)+'.p','rb'))
		V = pickle.load(open('calibration/V_frac1='+str(frac1)+'.p','rb'))
		calibrated = self.second_order_correction(values,M,V)

		out = open('calibration/'+outfilename,'w')
		out.write('\t'.join([x for x in header])+'\t'+'calibrated\n')
		for i in range(len(calibrated)):
			out.write(oldlines[i].strip()+'\t'+str(calibrated[i])+'\n')
		out.close()


	def learn_calibration(self,frac1,filename):
		truth,uncalibrated = self.read_file(filename)
		xtrain,ytrain = self.plot_uncalibrated_reliability(truth,uncalibrated,frac1)
		ir = self.isotonic_calibration(xtrain,ytrain)
		pickle.dump(ir,open('calibration/IR_frac1='+str(frac1)+'.p','wb'))
		M,V = self.interpolate_isotonic(ir)
		pickle.dump(M,open('calibration/M_frac1='+str(frac1)+'.p','wb'))
		pickle.dump(V,open('calibration/V_frac1='+str(frac1)+'.p','wb'))
		self.plot_calibrated_reliability(truth,uncalibrated,frac1,ir,M,V)


	def read_file(self,filename):
		file = open(filename,'r')
		f = file.read()
		file.close()
		f = f.strip().splitlines()
		truth = []
		uncalibrated = []
		for line in f:
			line = line.strip().split('\t')
			truth.append(int(line[0]))
			uncalibrated.append(float(line[1]))
		return truth, uncalibrated

	def plot_calibrated_reliability(self,truth,uncalibrated,frac1,ir,M,V):
		bins_0_sums = [float(i+.5)/self.numbins for i in range(self.numbins)]
		bins_0_lens = [1 for i in range(self.numbins)]
		bins_1_sums = [float(i+.5)/self.numbins for i in range(self.numbins)]
		bins_1_lens = [1 for i in range(self.numbins)]
		for i in range(len(truth)):
			val = self.second_order_correction([uncalibrated[i]],M,V)[0]
			bin = self.prob2interval(val)
			if truth[i] == 0:
				bins_0_sums[bin] += val
				bins_0_lens[bin] += 1
			elif truth[i] == 1:
				bins_1_sums[bin] += val
				bins_1_lens[bin] += 1
		xvals = []
		yvals = []
		total0 = sum(bins_0_lens)
		total1 = sum(bins_1_lens)
		newNs = float(frac1)*float(total0)/(1-frac1)
		downsample_scalar = float(newNs)/total1
		for bin in range(self.numbins):
			if bins_1_lens[bin] == 1 and bins_0_lens[bin] == 1:
				pass
			else:
				x = float(bins_0_sums[bin] + downsample_scalar*bins_1_sums[bin])/(bins_0_lens[bin] + downsample_scalar*bins_1_lens[bin])
				xvals.append(x)
				y = float(downsample_scalar*bins_1_lens[bin])/(bins_0_lens[bin]+downsample_scalar*bins_1_lens[bin])
				yvals.append(y)
		plt.plot(xvals,yvals,'o-')
		plt.plot([0,1],[0,1],':k')
		plt.xlabel('Calibrated Probability')
		plt.ylabel('Fraction with Label 1')
		plt.savefig('calibration/calibrated_reliability_frac1='+str(frac1)+'.pdf')
		plt.clf()		

	def plot_uncalibrated_reliability(self,truth,uncalibrated,frac1):
		#truth,uncalibrated = self.read_file(filename)
		xtrain = [] #for IR
		ytrain = [] #for IR
		bins_0_sums = [float(i+.5)/self.numbins for i in range(self.numbins)]
		bins_0_lens = [1 for i in range(self.numbins)]
		bins_1_sums = [float(i+.5)/self.numbins for i in range(self.numbins)]
		bins_1_lens = [1 for i in range(self.numbins)]
		for i in range(len(truth)):
			val = uncalibrated[i]
			xtrain.append(val)
			bin = self.prob2interval(val)
			#print bin
			if truth[i] == 0:
				bins_0_sums[bin] += val
				bins_0_lens[bin] += 1
			elif truth[i] == 1:
				bins_1_sums[bin] += val
				bins_1_lens[bin] += 1
		xvals = []
		yvals = []
		total0 = sum(bins_0_lens)
		total1 = sum(bins_1_lens)
		newNs = float(frac1)*float(total0)/(1-frac1)
		downsample_scalar = float(newNs)/total1
		for bin in range(self.numbins):
			if bins_1_lens[bin] == 1 and bins_0_lens[bin] == 1:
				pass
			else:
				x = float(bins_0_sums[bin] + downsample_scalar*bins_1_sums[bin])/(bins_0_lens[bin] + downsample_scalar*bins_1_lens[bin])
				xvals.append(x)
				y = float(downsample_scalar*bins_1_lens[bin])/(bins_0_lens[bin]+downsample_scalar*bins_1_lens[bin])
				yvals.append(y)
		plt.plot(xvals,yvals,'o-')
		plt.plot([0,1],[0,1],':k')
		plt.xlabel('Uncalibrated Probability')
		plt.ylabel('Fraction with Label 1')
		plt.savefig('calibration/uncalibrated_reliability_frac1='+str(frac1)+'.pdf')
		plt.clf()

		for x in xtrain:
			bin = self.prob2interval(x)
			ytrain.append(yvals[bin])
		return xtrain,ytrain
 
	def prob2interval(self,prob):
		a = float(prob)/(1.0/self.numbins)
		bin = int(a)
		if bin == self.numbins:
			bin = self.numbins-1
		return bin


	def isotonic_calibration(self,xtrain,ytrain):
		ir = IR(out_of_bounds='clip')
		ir.fit(xtrain,ytrain)
		#print ir
		return ir

	def interpolate_isotonic(self,ir,delta=1e-6):
		#learn the midpoints and the values of the intervals
		x = np.arange(0,1,0.001)
		y = ir.transform(x)
		if np.isnan(y[0]):
			y[0] = 0
		startpoints = [0]
		endpoints = []
		values = [y[0]]
		for i in range(1,len(x)):
			if abs(y[i]-values[-1]) > delta:
				startpoints.append(x[i])
				values.append(y[i])
				endpoints.append(x[i-1])
		endpoints.append(1)
		midpoints = []
		for i in range(len(startpoints)):
			midpoints.append(0.5*(endpoints[i]+startpoints[i]))
		midpoints.append(1)
		values.append(1)
		return midpoints,values

	def second_order_correction(self,cal_p_vec,M,V):
		newvals = []
		for cal_p in cal_p_vec:
			i = 0
			while cal_p > M[i]:
				i += 1
			if i == 0:
				newvals.append(0)
			else:
				leftpos = M[i-1]
				rightpos = M[i]
				leftval = V[i-1]
				rightval = V[i]
				pos = float(cal_p-leftpos)/(rightpos-leftpos)
				newval = leftval + (rightval-leftval)*pos
				newvals.append(newval)
		return newvals

#if __name__ == '__main__':
def main():
	import argparse
	parser = argparse.ArgumentParser()	
	parser.add_argument('--frac1',action='store',dest='frac1',default=0.1) #training set makeup: what percentage of training set is from label 1 (sweep)? required for both --learn and --apply
	parser.add_argument('--train',action='store_true',dest='learn')
	parser.add_argument('--input_train',action='store',dest='input_train') #file with training data, required for --learn (2 columns, no headers. first column is true values 0=neutral, 1=sweep. second column is uncalibrated probabilities)
	parser.add_argument('--apply',action='store_true',dest='apply')
	parser.add_argument('--input_apply',action='store',dest='input_apply') #file with any number of tab-delimited columns, with header line. One header line must be "uncalibrated". output will make a new file that adds an additional column called "calibrated"
	args = parser.parse_args()
	C = SmoothedIsotonicCalibration()

	if args.learn:
		C.learn_calibration(float(args.frac1),args.input_train)
	elif args.apply:
		C.apply_calibration(float(args.frac1),args.input_apply)
