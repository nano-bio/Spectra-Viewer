from __future__ import division
from warnings import catch_warnings, simplefilter
from h5py import File
from numpy import pi, sqrt, exp, array, zeros, mean, arange, append, log, median, std, where
from numpy import abs as npabs
from numpy import sum as npsum
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.ticker import MultipleLocator, FormatStrFormatter



font = 12

from matplotlib import rcParams
rcParams['xtick.labelsize']=font
rcParams['ytick.labelsize']=font
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

class WlError(Exception):
    pass

def lorentzian(x,*p):
	func = 1
	for i in range(len(p)//3):
		gamma = p[2*i+0]
		x0 = p[2*i+1]
		A = p[2*i+2]
		func -= 1/(pi*gamma) * (gamma**2/((x-x0)**2 + gamma**2)) * A# + A * 8.0147/100 * 1/(pi*gamma) * (gamma**2/((x-(x0-1))**2 + gamma**2))
	return func

def gaussian(x,*p):
	func = 1
	for i in range(len(p)//3):
		sigma = p[2*i+0]
		x0 = p[2*i+1]
		A = p[2*i+2]
		func -= 1/sqrt(2*pi*sigma**2) * exp(-((x-x0)**2)/(2*sigma**2)) * A# + A * 8.0147/100 * 1/(pi*gamma) * (gamma**2/((x-(x0-1))**2 + gamma**2))
	return func

def is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

def find_nearest(a, a0):
	"Element in nd array `a` closest to the scalar value `a0`"
	a0 = array(a0)
	s = a0.size
	idx = zeros(s)
	if s == 1:
		idx = npabs(a - a0).argmin()
		return idx
	elif s > 1:
		for i in range(s):
			idx[i] = npabs(a - a0[i]).argmin()
		return idx

def getfile(filename):
	f = h5py.File(filename, 'r')
	mset = array(f[u'FullSpectra/MassAxis'])
	peakdata = array(f[u'PeakData/PeakData'])
	#timedata = array(f[u'TimingData/BufTimes'])
	#peaktable = array(f[u'PeakData/PeakTable'])
	aqlog = f[u'AcquisitionLog/Log']
	return mset,peakdata,aqlog

def array_slicer(array,ind0s,indlen):
	out_array = zeros([len(ind0s),array.shape[1]])
	for i,ind0 in enumerate(ind0s):
		out_array[i,:] = mean(array[ind0:int(ind0+indlen),:],axis=0)
	return out_array

class spectra:
	def __init__(self,filename):
		f = File(filename, 'r')
		try:
			self.numadds = f.attrs["NbrBlocks"][0]*f.attrs["NbrCubes"][0]*f.attrs["NbrMemories"][0]*f.attrs["NbrWaveforms"][0]
			self._readadds = True
		except KeyError:
			self.numadds = 1
			self._readadds = False
		#print("nblocks",f.attrs["NbrBlocks"][0]*f.attrs["NbrCubes"][0]*f.attrs["NbrMemories"][0])
		#for item in f.attrs.keys():
		#    print(item + ":", f.attrs[item])
		#self.config = str(f.attrs["Configuration File Contents"])
		#print(sys.stdout.buffer.write(f.attrs["Configuration File Contents"]))
		#print(f[u'/'].keys())
		try:
			self.mset = array(f[u'FullSpectra/MassAxis'])
			self.dset = array(f[u'FullSpectra/SumSpectrum'])
			self.peakdata = array(f[u'PeakData/PeakData'])
			self.aqlog = f[u'AcquisitionLog/Log']
			peaktable = array(f[u'PeakData/PeakTable'])
			self.peak_ranges = [list(i)[2:] for i in peaktable]
			self.read_error = False
		except KeyError:
			self.read_error = True
		
		self.wl_guess = 0
		self.parent_peak = 266
		self.tag_mass = 4
		#powcor = np.loadtxt("power_master.txt")
		#self.pcwl = powcor[:,1]
		#self.pcfac = npabs(powcor[:,2]/np.max(powcor[:,2]))
		
	def read_wl_log(self):
		self.log_times = []
		self.wls = []
		real_times = []
		wl_diffs = 0
		
		for aq in self.aqlog[:]:
			self.log_times.append(aq[0])
			real_times.append(False)
			
			if is_number(aq[-1]): 
				self.wls.append(float(aq[-1]))
				real_times[-1] = True
		if len(self.wls) > 0:
			self.wls = array(self.wls)
			self.med_wl_diff = median(self.wls[1:] - self.wls[:-1])
			
			self.log_times = array(self.log_times)
			self.log_times -= self.log_times[0]
			self.log_inds = (self.log_times * self.peakdata.shape[0]) // self.log_times[-1]
			
			self.wl_inds = self.log_inds[real_times]
			self.med_stepsize = round(median(self.log_inds[1:]-self.log_inds[:-1]))
			if self.med_stepsize == 0:
				self.med_stepsize = 1
		else:
			raise WlError("No wavelengths found in log.")

	def make_wl_log(self,wl0,steps=None,channels_per_step=1,stepsize=0.01):
		if steps == None:
			steps = self.peakdata.shape[0]
		self.log_inds = []
		self.wls = []
		for i in range(steps):
			self.wls.append(wl0+i*stepsize)
			self.log_inds.append(channels_per_step*i)
		#self.log_inds.append(steps*channels_per_step)
		self.log_inds = array(self.log_inds)
		self.wl_inds = self.log_inds
		self.med_stepsize = channels_per_step

	def gen_data(self):
		num_bg_meas = self.peakdata[:,:,:,:].shape[2]-1
		channels_per_step = self.med_stepsize
		
		self.raw_signal = npsum(self.peakdata[:,:,0,:],axis=1)*self.numadds
		self.raw_background = npsum(npsum(self.peakdata[:,:,1:,:],axis=2),axis=1)*self.numadds
		
		out_array = zeros([len(self.wl_inds),self.raw_signal.shape[1]])
		bg_out_array = zeros([len(self.wl_inds),self.raw_background.shape[1]])
		
		for i,ind0 in enumerate(self.wl_inds):
			out_array[i,:] = npsum(self.raw_signal[ind0:int(ind0+channels_per_step),:],axis=0)
			bg_out_array[i,:] = npsum(self.raw_background[ind0:int(ind0+channels_per_step),:],axis=0)
			
		
		self.signal = out_array
		self.background = bg_out_array
		
		signal_err = sqrt(self.signal)
		background_err = sqrt(self.background)
		
		with catch_warnings():
			simplefilter("ignore",category=RuntimeWarning)
			#print(self.signal.shape,(self.background/num_bg_meas).shape)
			self.diff_signal = self.signal - (self.background/num_bg_meas)
			self.flat_signal = self.signal/(self.background/num_bg_meas)
			self.flat_signal_err = sqrt(signal_err**2*(num_bg_meas/self.background)**2+background_err**2*(num_bg_meas*self.signal/self.background**2)**2)
		
	
	
	def plot_peakdatasmooth(self,pdsax,peak,wl=5,po=2,cal=False,flat=False,diff=False):
		#self.pdcspfig,self.pdcspax = plt.subplots(nrows=1,ncols=1)
		if not cal:
			if flat:
				self.pds_matrix = savgol_filter(self.flat_signal[:,peak],window_length=wl,polyorder=po,axis=0)
				pdsax.ax.set_ylabel("Relative Ion Yield",fontsize=font)
			elif diff:
				self.pds_matrix = savgol_filter(self.diff_signal[:,peak],window_length=wl,polyorder=po,axis=0)
				pdsax.ax.set_ylabel("Differential Ion Yield (counts)",fontsize=font)
			else:
				self.pds_matrix = savgol_filter(self.signal[:,peak],window_length=wl,polyorder=po,axis=0)
				pdsax.ax.set_ylabel("Raw Ion Yield (counts)",fontsize=font)
			pdsax.ax.plot(self.pds_matrix)
			pdsax.ax.set_xlabel("Bin index",fontsize=font)
		else:
			if flat:
				self.pds_matrix = savgol_filter(self.flat_signal[:,peak],window_length=wl,polyorder=po,axis=0)
				pdsax.ax.set_ylabel("Relative Ion Yield",fontsize=font)
			elif diff:
				self.pds_matrix = savgol_filter(self.diff_signal[:,peak],window_length=wl,polyorder=po,axis=0)
				self.pds_matrix_sum = savgol_filter(npsum(self.diff_signal[:,peak],axis=1),window_length=wl,polyorder=po,axis=0)
				pdsax.ax.set_ylabel("Differential Ion Yield (counts)",fontsize=font)
			else:
				self.pds_matrix = savgol_filter(self.signal[:,peak],window_length=wl,polyorder=po,axis=0)
				pdsax.ax.set_ylabel("Raw Ion Yield (counts)",fontsize=font)
			pdsax.ax.plot(self.wls,self.pds_matrix)
			if diff:
				pdsax.ax.plot(self.wls,self.pds_matrix_sum,label="Sum of curves")
			pdsax.ax.set_xlabel("Wavelength (nm)",fontsize=font)
		
		pdsax.ax.minorticks_on()
		
	
	def find_lines(self,bsax,num_bins=1,thresh=5):
		#self.flfig,self.flfig = plt.subplots(nrows=1,ncols=1)
		
		smoothed = savgol_filter(self.flat_signal,window_length=15,polyorder=1,axis=0)
		
		with catch_warnings():
			simplefilter("ignore",category=RuntimeWarning)
			noise_levels = std(self.flat_signal - smoothed,axis=0)
			abs_scatter = npabs(-log(self.flat_signal))
		
		self.mass_channel = arange(abs_scatter.shape[1])+1
		self.peak_counts = zeros(abs_scatter.shape[1])
		
		mean_level = mean(self.raw_signal,axis=0)
				
		for i in range(abs_scatter.shape[1]):
			if mean_level[i] > 50:
				ww = where(abs_scatter[:,i] > (noise_levels[i]*thresh))[0]
				self.peak_counts[i] = len(ww)
		
		bsax.ax.bar(self.mass_channel,self.peak_counts,width=1)
		
		#self.flfig2.plot(self.wls,self.flat_signal[:,peak-1],label=peak)
		#self.flfig2.plot(self.wls,self.flat_signal[:,peak-1]-smoothed[:,peak-1],label="Noise")
		#self.flfig2.set_xlabel("Wavelength (nm)")
		#self.flfig2.set_ylabel("Ion Yield")
		#self.flfig2.legend()
		
		bsax.ax.set_xlabel("Mass",fontsize=font)
		bsax.ax.set_ylabel("Channels",fontsize=font)
		bsax.ax.minorticks_on()
			
	def plot_ms(self,msax):
		msax.ax.plot(self.mset,self.dset)
		msax.ax.set_xlabel("Mass per Charge (u/e)",fontsize=font)
		msax.ax.set_ylabel("Yield (counts)",fontsize=font)
		msax.ax.minorticks_on()
			
	
if __name__ == "__main__":
	#path = "/Volumes/Storage/ClusToF/CsHeH/"
	path = "/Volumes/Storage/ClusToF/Benz_a_anthracene/"
	#path = "/Volumes/KINGSTON/CsHeH/"
	#path = "/Users/michael/Downloads/"
	masses = arange(266,430,1)
	#masses = append(masses,array([256,260,264,300]))
	savecorr = False
	
	
	if 0:
		spec1 = spectra(path+"DataFile_2019.10.10-14h56m03s_AS.h5") # 5723 BAA 400-500nm  0,1nm /60s
		spec1.read_wl_log()
		#spec1.make_wl_log(wl0=400,stepsize=0.1)
		spec1.gen_data()
		#spec1.plot_peakdata(array([228]),cal=True)
		spec1.plot_peakdatasmooth(array([63,114,202,226,228,456,685]),wl=9,po=0,cal=True,flat=True)
		spec1.find_lines(thresh=5)
		
	if 0:
		spec2 = spectra(path+"DataFile_2019.10.09-20h04m03s_AS.h5") # 5723 BAA 400-500nm  0,1nm /60s
		spec2.read_wl_log()
		#spec2.make_wl_log(wl0=400,stepsize=0.1)
		spec2.gen_data()
		#spec2.plot_peakdata(array([228]),cal=True)
		spec2.plot_peakdatasmooth(array([63,114,202,226,228,456,685]),wl=9,po=0,cal=True,flat=True)
		spec2.find_lines(thresh=5)
		
	plt.show()