
import os

try:
    from . import ImageProcessing as IP
except:
    import ImageProcessing as IP

def handle_data_01():
    SEP = '\\'
    folder = os.path.expanduser(r'~\Downloads\train' + SEP)
    paths = os.listdir(folder)
    cat_paths = [folder+p for p in paths if p[:3] == 'cat']
    dog_paths = [folder+p for p in paths if p[:3] == 'dog']

    for i, (cpath, dpath) in enumerate(zip(cat_paths, dog_paths)):
        cimg, dimg = IP.load_img(cpath), IP.load_img(dpath)
        cimg, dimg = IP.set_square(cimg), IP.set_square(dimg)
        cpath, dpath = str(IP.THISPATH / f"Images\\Cats\\{i}.jpg"), str(IP.THISPATH / f"Images\\Dogs\\{i}.jpg")
        IP.save_img(cimg, cpath)
        IP.save_img(dimg, dpath)

if __name__ == "__main__":
    pass
