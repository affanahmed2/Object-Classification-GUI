import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

sys.path.append("scripts/")
sys.path.append("imports/")

from UI import Ui_MainWindow




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # set up the user interface from Designer.
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)







if __name__ != "__main__":
    exit()
    
    
    
    
    
    
    
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
