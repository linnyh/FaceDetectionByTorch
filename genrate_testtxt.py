import os


if __name__ == '__main__':
    fp = open('./CelebA/Anno/list_attr_celeba.txt')
    for line in fp.readlines()[1002:2002]:
        with open(r'./test1000.txt',"a") as f:
            f.write(line)