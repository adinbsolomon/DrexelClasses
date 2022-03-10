
import os

if __name__ == "__main__":

    assert (os.path.basename(os.path.abspath("")) == "CS-615-Final"), "Make sure you're in the Final Project repo"

    print("\nPip Installs")
    os.system("pip install tqdm opencv-python")
    
    print("\nCustom Installs and Cleanup")
    os.system("git clone https://github.com/AdinSolomon/MyUtils.git")
    os.chdir("MyUtils")
    os.system("git checkout AdinsUtils")
    os.chdir("..")
    os.system("move MyUtils/MyPython MyPython")
    os.system("rmdir /Q /S MyUtils")
