import os

mydir = r'./CelebA/Img/img_align_celeba'
datatxt = r'./CelebA/Anno/list_attr_celeba.txt'

if __name__ == '__main__':
    fp = open(datatxt)
    for file in fp.readlines()[2:5002]:
        with open(r"./train5000.txt", "a") as f:
            f.write(file)
    '''
    for pathstr,dirlist,filelist in os.walk(mydir):
        for file in filelist:
            with open(r"./train_all.txt","a") as f:
                f.write(file+'\n')

    for pathstr, dirlist, filelist in os.walk(mydir):
        for file in filelist[:10000]:
            with open(r"./train10000.txt", "a") as f:
                f.write(file + '\n')
    '''
