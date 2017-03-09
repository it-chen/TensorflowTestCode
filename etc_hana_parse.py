import struct
import numpy as np
import os
from PIL import Image

sz_record = 8199


def read_record_ETL8G(f):
    s = f.read(sz_record)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)


def read_hiragana():
    # Character type = 72, person = 160, y = 127, x = 128
    ary = np.zeros([72, 160, 127, 128], dtype=np.uint8)

    for j in range(1, 33):
        filename = 'C:/Users/lele.chen/Downloads/ETL8G/ETL8G/ETL8G_{:02d}'.format(j)
        with open(filename, 'rb') as f:

            for id_dataset in range(5):
                f.seek(id_dataset * 956 * sz_record)
                moji = 0
                for i in range(956):
                    r = read_record_ETL8G(f)
                    iE = Image.eval(r[-1], lambda x: 255 - x * 16)
                    forder = 'C:/data/ETL8G/{:s}'.format(r[2].decode('utf8').strip())
                    if b'.HIRA' in r[2]:
                        if not os.path.exists(forder):
                            os.mkdir(forder)
                        fn = 'C:/data/ETL8G/{:s}/ETL8G_{:d}_{:s}.png'.format(r[2].decode('utf8').strip(),r[0], hex(r[1])[-4:])
                        iE.save(fn, 'PNG')
                        print(fn)

read_hiragana()