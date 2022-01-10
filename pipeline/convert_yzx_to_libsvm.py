#!/usr/bin/python
import sys
import argparse

def read_yzx(finame):
    result = []
    fi = open(finame, 'r')
    for item in fi:
        ll = item.strip('\n').split(' ')
        i = 0
        kv = {}
        line = []
        for l in ll:
            #process y(label)
            if i == 0:
                line.append(l[0])
                i+=1
                continue
            #skip z (market price)
            if i == 1 :
                i+=1
                continue
            #process x
            k, v = l.split(':')
            kv[int(k)] = v
            i+=1
        for k in sorted(kv):
            line.append(str(k)+':'+kv[k])
        result.append(line)
    return result

def write_libsvm(libsvm_file, samples):
    fo = open(libsvm_file, 'w')
    for line in samples:
        i = 0
        for c in line:
            i+=1
            fo.write(c) 
            if i < len(line): 
                fo.write(' ')
            else:
                fo.write('\n')

def main():
    parser = argparse.ArgumentParser(description='please input configre file')
    parser.add_argument('--yzx_file', type=str, default="train.yzx.txt")
    parser.add_argument('--libsvm_file', type=str, default="train_libsvm.txt")
    args = parser.parse_args()
    yzx_file = args.yzx_file
    libsvm_file = args.libsvm_file
    read_yzx(yzx_file)
    write_libsvm(libsvm_file)

if __name__ == '__main__':
    main()
