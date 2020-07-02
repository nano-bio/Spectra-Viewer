from sys import argv, exit
from numpy import array, append

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QAction, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSizePolicy, QPushButton, QGridLayout, QFileDialog
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtCore import QThread

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from Spectra_Class import spectra, WlError
from Spectra_UI import Ui_MainWindow

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar



class StandardItemBin(QStandardItem):
	def __lt__(self, other):
		return float(self.text().strip('Bin ')) < float(other.text().strip('Bin '))

class StandardItem(QStandardItem):
	def __lt__(self, other):
		return float(self.text()) < float(other.text())

class StandardItemMass(QStandardItem):
	def __lt__(self, other):
		return int(self.text()[5:]) < int(other.text()[5:])


class MyWindow(QMainWindow,Ui_MainWindow):
	def __init__(self,parent=None):
		#super().__init__()
		super(MyWindow, self).__init__(parent)
		#uic.loadUi('pyqt5test_ui_191024.ui', self)
		#self.ui = Ui_MainWindow()
		self.setupUi(self)
		
		"""
		Make plot widget
		"""
		self.ms_tab = self._WidgetPlotfunc(self.widget_mpl_ms)
		self.bs_tab = self._WidgetPlotfunc(self.widget_mpl_bs)
		self.iy_tab = self._WidgetPlotfunc(self.widget_mpl_iy)
		
		
		
		self.lineEdit_filename.setReadOnly(True)
		self.file_loaded = False
		
		
		extractAction = QAction("Open h5-file", self)
		extractAction.setShortcut("Ctrl+O")
		extractAction.triggered.connect(self.load_file)
		
		exportAction = QAction("Export Mass Spectrum", self)
		exportAction.triggered.connect(self.export_mass_spec)
		
		self.pushButton_loadh5.clicked.connect(self.load_file)
		self.menuFile.addAction(extractAction)
		self.menuFile.addSeparator()
		self.menuFile.addAction(exportAction)
		
		self.comboBox_wl.setIndex = 0
		self.check_list = []
		self.flat_bool = True
		self.diff_bool = False
		self.doubleSpinBox_wl0.setReadOnly(True)
		self.doubleSpinBox_wlstep.setReadOnly(True)
		
		self.bs_thresh_button.clicked.connect(lambda: self.draw_bs())
		self.smooth_spinBox.valueChanged.connect(lambda: self.smooth_box_changed())
		
		#self.iy_flattened_box.stateChanged.connect(lambda: self.draw_pds())
		
		self.comboBox_flat.currentIndexChanged.connect(lambda: self.flat_box_changed())
		self.iy_calib_box.stateChanged.connect(lambda: self.cal_box_changed())
		self.power_corr_box.stateChanged.connect(lambda: self.draw_pds())
		self.comboBox_wl.currentIndexChanged.connect(lambda: self.wls_box_changed())
		self.gen_wls_Button.clicked.connect(lambda: self.gen_wls())
		self.pushButton_export.clicked.connect(lambda: self.export_ascii())
		self.pushButton_bincreate.clicked.connect(lambda: self.add_bin())
		
	
		
	def load_file(self):
		openFileName = QFileDialog.getOpenFileName(self,"Select File(s)","","*.h5;*.hf5")
		if openFileName[0]:
			self.filename = openFileName[0]
			self.thread = Worker(self.filename)
			#self.thread.started.connect(self.thread.run)
			self.thread.finished.connect(lambda: self.load_finish(self.thread.spec_obj))
			self.thread.start()
			self.statusBar().showMessage('Loading File')
			
	def wls_box_changed(self):
		if self.comboBox_wl.currentIndex() == 0:
			self.read_wls()
			self.gen_wls_Button.setEnabled(False)
			self.doubleSpinBox_wl0.setReadOnly(True)
			self.doubleSpinBox_wlstep.setReadOnly(True)
		elif self.comboBox_wl.currentIndex() == 1:
			self.select_gen_wls()
	
	def flat_box_changed(self):
		if self.comboBox_flat.currentIndex() == 0:
			self.flat_bool = True
			self.diff_bool = False
			self.cal_box_changed()
		elif self.comboBox_flat.currentIndex() == 1:
			self.flat_bool = False
			self.diff_bool = True
			self.cal_box_changed()
		elif self.comboBox_flat.currentIndex() == 2:
			self.flat_bool = False
			self.diff_bool = False
			self.cal_box_changed()
	
	def smooth_box_changed(self):
		if (self.smooth_spinBox.value() % 2) != 0: 
			self.draw_pds()
		else:
			self.smooth_spinBox.setValue(self.smooth_spinBox.value() + 1)
			
	def select_gen_wls(self):
		self.gen_wls_Button.setEnabled(True)
		self.doubleSpinBox_wl0.setReadOnly(False)
		self.doubleSpinBox_wlstep.setReadOnly(False)
	
	def gen_wls(self):
		if self.file_loaded:
			self.spec_obj.make_wl_log(wl0=self.doubleSpinBox_wl0.value(),stepsize=self.doubleSpinBox_wlstep.value())
			self.spec_obj.gen_data()
			self.draw_bs()
		
	def read_wls(self):
		self.spec_obj.read_wl_log()
		self.show_read_wls()
		self.comboBox_wl.setEnabled(True)
		self.spec_obj.gen_data()
		self.draw_bs()
		
	def cal_box_changed(self):
		if ((not self.iy_calib_box.isChecked()) or self.comboBox_flat.currentIndex() != 0):
			self.power_corr_box.setEnabled(False)
		else:
			self.power_corr_box.setEnabled(True)
		self.draw_pds()
	
	def _cannot_read_wls(self):
		self.comboBox_wl.setCurrentIndex(1)
		self.comboBox_wl.setEnabled(False)
		self.select_gen_wls()
		if not self.spec_obj._readadds:
			self.statusBar().showMessage('Wavelength data not found in log, manual input required! No co-adding information found in file, unable to determine absolute statistical uncertanties.')
		else:
			self.statusBar().showMessage('Wavelength data not found in log, manual input required!')
		self.reset_plots()
			
	def load_finish(self,specobj):
		self.spec_obj = specobj
		if not self.spec_obj.read_error:
			self.file_loaded = True
			self.lineEdit_filename.setText(self.filename)
			if not self.spec_obj._readadds:
				self.statusBar().showMessage('Loading Complete. No co-adding information found in file, unable to determine absolute statistical uncertanties.')
			else:
				self.statusBar().showMessage('Loading Complete')
			try:
				self.read_wls()
			except WlError:
				self._cannot_read_wls()
			self.draw_ms()
		else:
			self.statusBar().showMessage('Error Loading File. Input file not compatible with this software.')
	
	def draw_ms(self):
		self.ms_tab.ax.clear()
		self.spec_obj.plot_ms(self.ms_tab)
		self.ms_tab.ax.set_ylim([0,None])
		self.ms_tab.draw()
	
	def draw_bs(self):
		if self.file_loaded:
			self.reset_plots()
			self.spec_obj.find_lines(self.bs_tab,thresh=float(self.bs_thresh_text.value()))
			self.bs_tab.draw()
			
			self.tableModel = QStandardItemModel(self)
			self.tableModel.clear()
			self.tableModel.setHorizontalHeaderLabels(["Channel ID","Mass Channel","Brightness","Lower Limit","Upper Limit"])
			items = []
			for i in range(len(self.spec_obj.mass_channel)):
				col0 = StandardItemBin("%s"%self.spec_obj.mass_channel_name[i].decode())
				col1 = StandardItem("%i"%self.spec_obj.mass_channel[i])
				col2 = StandardItem("%i"%self.spec_obj.peak_counts[i])
				col3 = StandardItem("%g"%self.spec_obj.peak_ranges[i][0])
				col4 = StandardItem("%g"%self.spec_obj.peak_ranges[i][1])
				items.append([col0,col1,col2,col3,col4])
				items[i][0].setCheckable(True)
				self.tableModel.appendRow(items[i])
			self.tableView.setModel(self.tableModel)
			self.tableView.setSortingEnabled(True)
			self.tableModel.itemChanged.connect(lambda: self.checked(items))
			self.pushButton_deselect.clicked.connect(lambda: self.decheck(items))
		
	def draw_pds(self):
		if self.file_loaded:
			self.iy_tab.ax.clear()
			if len(self.check_list) > 0:
				self.spec_obj.plot_peakdatasmooth(self.iy_tab,array(self.check_list),wl=int(self.smooth_spinBox.value()),po=0,cal=self.iy_calib_box.isChecked(),flat=self.flat_bool,diff=self.diff_bool,pow_corr=self.power_corr_box.isChecked())
				if self.diff_bool:
					self.iy_tab.ax.legend([*self.leg_list,"Sum"],loc=0,title="Mass channel")
				else:
					self.iy_tab.ax.legend(self.leg_list,loc=0,title="Mass channel")
			self.iy_tab.draw()
	
	def show_read_wls(self):
		self.doubleSpinBox_wl0.setReadOnly(True)
		self.doubleSpinBox_wl0.setValue(self.spec_obj.wls[0])
		self.doubleSpinBox_wlstep.setReadOnly(True)
		self.doubleSpinBox_wlstep.setValue(self.spec_obj.med_wl_diff)
	
	def decheck(self,items_list):
		self.check_list = []
		for i in range(len(items_list)):
			items_list[i][0].setCheckState(0)		
			
	def checked(self,items_list):
		self.check_list = []
		self.leg_list = []
		for i in range(len(items_list)):
			if items_list[i][0].checkState() == 2:
				self.check_list.append(int(i))
				self.leg_list.append("%g - %g"%(self.spec_obj.peak_ranges[i][0],self.spec_obj.peak_ranges[i][1]))
		self.draw_pds()
	
	def add_bin(self):
		if self.file_loaded:
			self.spec_obj.add_bin(int(self.binstart_spinBox.value()),int(self.binwidth_spinBox.value()),int(self.binsep_spinBox.value()),int(self.binsteps_spinBox.value()))
			self.draw_bs()
	
	def _WidgetPlotfunc(self,mpl_qwidget):
		mpl_qwidget.setLayout(QVBoxLayout())
		canvas = PlotCanvas(mpl_qwidget)
		toolbar = NavigationToolbar(canvas,mpl_qwidget)
		mpl_qwidget.layout().addWidget(toolbar)
		mpl_qwidget.layout().addWidget(canvas)
		return canvas
	
	
	def reset_plots(self):
		self.bs_tab.ax.clear()
		self.bs_tab.draw()
		self.iy_tab.ax.clear()
		self.iy_tab.draw()
	
	def export_mass_spec(self):
		saveFileName = QFileDialog.getSaveFileName(self,"Export File","","ASCII file (*.dat *.txt)")
		if saveFileName[0]:
			header = "#%15s\t%10s\n"%("m/z","Counts")
			with open(saveFileName[0],'w') as f:
				#f.write(header)
				for i in range(len(self.spec_obj.mset)):
					f.write("%16.10f\t%10i\n"%(self.spec_obj.mset[i],self.spec_obj.dset[i]))
		self.statusBar().showMessage('File export complete')
				
			
	
	def export_ascii(self):
		saveFileName = QFileDialog.getSaveFileName(self,"Export File","","ASCII file (*.dat *.txt);;CSV file (*.csv)")
		if saveFileName[0]:
			if saveFileName[1] == "ASCII file (*.dat *.txt)": separator = " "
			elif saveFileName[1] == "CSV file (*.csv)": separator = ","
			if self.iy_calib_box.isChecked():
				header = "#%18s%1s"%("Wavelengths_(nm)",separator)
				for cl in self.check_list:
					header += "%19s%1s"%("mass%g-%g"%(self.spec_obj.peak_ranges[cl][0],self.spec_obj.peak_ranges[cl][1]),separator)
				header = header[:-1] +"\n"
				if (self.power_corr_box.isChecked() and self.comboBox_flat.currentIndex() == 0):
					out_table = append(array(self.spec_obj.wls_pow)[...,None],self.spec_obj.pds_matrix,1)
				else:
					out_table = append(array(self.spec_obj.wls)[...,None],self.spec_obj.pds_matrix,1)
			else:
				header = "#"
				for cl in self.check_list:
					header += "%24s"%("mass%g-%g"%(self.spec_obj.peak_ranges[cl][0],self.spec_obj.peak_ranges[cl][1]))
				header +="\n"
				out_table = self.spec_obj.pds_matrix
			with open(saveFileName[0],'w') as f:
				f.write(header)
				for r in range(out_table.shape[0]):
					for c in range(out_table.shape[1]-1):
						f.write("%19g%s"%(out_table[r,c],separator))
					f.write("%19g\n"%(out_table[r,out_table.shape[1]-1]))
			self.statusBar().showMessage('File export complete')
			
							

class Worker(QThread):
	def __init__(self, filepath):
		super().__init__()
		self.fp = filepath
	def run(self):
		self.spec_obj = spectra(self.fp)
		#self.finished.emit(self.spec_obj)
		
class PlotCanvas(FigureCanvas):
	def __init__(self, parent=None, width=10, height=8, dpi=100):
		fig = Figure(figsize=(width, height), dpi=dpi)
		FigureCanvas.__init__(self, fig)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.plot()

	def plot(self):
		self.ax = self.figure.add_subplot(111)

		
if __name__ == '__main__':

	app = QApplication(argv)
	window = MyWindow()
	window.show()
	exit(app.exec_())