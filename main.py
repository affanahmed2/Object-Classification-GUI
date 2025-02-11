import sys
from PyQt5.QtWidgets import QApplication


sys.path.append("scripts/")

from UI import MainWindow



def CreateWindow():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.trainClicked.connect(train_function) 
    window.show()
    sys.exit(app.exec_())
    
    
    

def train_function():
    # Placeholder function that does nothing.
    print("Train button pressed!")
    
    
if __name__ == '__main__':
    CreateWindow()