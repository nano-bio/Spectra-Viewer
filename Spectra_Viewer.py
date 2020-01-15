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
		
		self.pushButton_loadh5.clicked.connect(self.load_file)
		self.menuFile.addAction(extractAction)
		
		self.comboBox_wl.setIndex = 0
		self.check_list = []
		self.flat_bool = True
		self.diff_bool = False
		self.doubleSpinBox_wl0.setReadOnly(True)
		self.doubleSpinBox_wlstep.setReadOnly(True)
		
		self.bs_thresh_button.clicked.connect(lambda: self.draw_bs())
		self.smooth_spinBox.valueChanged.connect(lambda: self.draw_pds())
		
		#self.iy_flattened_box.stateChanged.connect(lambda: self.draw_pds())
		
		self.comboBox_flat.currentIndexChanged.connect(lambda: self.flat_box_changed())
		self.iy_calib_box.stateChanged.connect(lambda: self.draw_pds())
		self.comboBox_wl.currentIndexChanged.connect(lambda: self.wls_box_changed())
		self.gen_wls_Button.clicked.connect(lambda: self.gen_wls())
		self.pushButton_export.clicked.connect(lambda: self.export_ascii())
		
		
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
		elif self.comboBox_flat.currentIndex() == 1:
			self.flat_bool = False
			self.diff_bool = True
		elif self.comboBox_flat.currentIndex() == 2:
			self.flat_bool = False
			self.diff_bool = False
		self.draw_pds()
			
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
		
		
	def _cannot_read_wls(self):
		self.comboBox_wl.setCurrentIndex(1)
		self.comboBox_wl.setEnabled(False)
		self.select_gen_wls()
		self.statusBar().showMessage('Wavelength data not found in log, manual input required!')
		self.reset_plots()
			
	def load_finish(self,specobj):
		self.file_loaded = True
		self.spec_obj = specobj
		self.lineEdit_filename.setText(self.filename)
		self.statusBar().showMessage('Loading Complete')
		try:
			self.read_wls()
		except WlError:
			self._cannot_read_wls()
		self.draw_ms()
		
	
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
			self.tableModel.setHorizontalHeaderLabels(["Mass Channel","Brightness","Lower Limit","Upper Limit"])
			items = []
			for i in range(len(self.spec_obj.mass_channel)):
				col1 = StandardItem("%i"%self.spec_obj.mass_channel[i])
				col2 = StandardItem("%i"%self.spec_obj.peak_counts[i])
				col3 = StandardItem("%g"%self.spec_obj.peak_ranges[i][0])
				col4 = StandardItem("%g"%self.spec_obj.peak_ranges[i][1])
				items.append([col1,col2,col3,col4])
				items[i][0].setCheckable(True)
				self.tableModel.appendRow(items[i])
			self.tableView.setModel(self.tableModel)
			self.tableView.setSortingEnabled(True)
			self.tableModel.itemChanged.connect(lambda: self.checked(items))
		
	def draw_pds(self):
		if self.file_loaded:
			self.iy_tab.ax.clear()
			if len(self.check_list) > 0:
				self.spec_obj.plot_peakdatasmooth(self.iy_tab,array(self.check_list),wl=int(self.smooth_spinBox.value()),po=0,cal=self.iy_calib_box.isChecked(),flat=self.flat_bool,diff=self.diff_bool)
			self.iy_tab.draw()
	
	def show_read_wls(self):
		self.doubleSpinBox_wl0.setReadOnly(True)
		self.doubleSpinBox_wl0.setValue(self.spec_obj.wls[0])
		self.doubleSpinBox_wlstep.setReadOnly(True)
		self.doubleSpinBox_wlstep.setValue(self.spec_obj.med_wl_diff)
			
	def checked(self,items_list):
		self.check_list = []
		for i in range(len(items_list)):
			if items_list[i][0].checkState() == 2:
				self.check_list.append(int(i))
		self.draw_pds()
	
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
	
	def export_ascii(self):
		saveFileName = QFileDialog.getSaveFileName(self,"Select File","","ASCII file (*.dat *.txt);;CSV file (*.csv)")
		if saveFileName[0]:
			if saveFileName[1] == "ASCII file (*.dat *.txt)": separator = " "
			elif saveFileName[1] == "CSV file (*.csv)": separator = ","
			if self.iy_calib_box.isChecked():
				header = "#%18s%1s"%("Wavelengths_(nm)",separator)
				for cl in self.check_list:
					header += "%19s%1s"%("mass%i"%cl,separator)
				header = header[:-1] +"\n"
				out_table = append(self.spec_obj.wls[...,None],self.spec_obj.pds_matrix,1)
			else:
				header = "#"
				for cl in self.check_list:
					header += "mass%20i"%cl
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