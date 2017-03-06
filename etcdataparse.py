import struct
import numpy as np
from PIL import Image

sz_record = 8199


def read_record_ETL8G(f):
    s = f.read(sz_record)
    r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
    iF = Image.frombytes('F', (128, 127), r[14], 'bit', 4)
    iL = iF.convert('L')
    return r + (iL,)

id_record = 0

# for j in range(1, 33):
#     filename = 'C:/Users/lele.chen/Downloads/ETL8G/ETL8G/ETL8G_{:02d}'.format(j)
#     id_record = 0
#
#     with open(filename, 'r') as f:
#         f.seek(id_record * sz_record)
#         r = read_record_ETL8G(f)
#
#     print(r[0:-2], hex(r[1]))
#     iE = Image.eval(r[-1], lambda x: 255-x*16)
#     fn = 'ETL8G_{:d}_{:s}.png'.format((r[0]-1)%20+1, hex(r[1])[-4:])
#     iE.save(fn, 'PNG')


def read_hiragana():
    # Character type = 72, person = 160, y = 127, x = 128
    ary = np.zeros([72, 160, 127, 128], dtype=np.uint8)

    for j in range(1, 33):
        filename = 'C:/Users/lele.chen/Downloads/ETL8G/ETL8G/ETL8G_{:02d}'.format(j)
        with open(filename, 'rb') as f:
            for id_dataset in range(5):
                moji = 0
                for i in range(956):
                    r = read_record_ETL8G(f)

                    iE = Image.eval(r[-1], lambda x: 255-x*16)
                    fn = 'ETL8G_{:d}_{:s}.png'.format((r[0]-1)%20+1, hex(r[1])[-4:])
                    iE.save(fn, 'PNG')

                    # if b'.HIRA' in r[2]:
                    #     ary[moji, (j - 1) * 5 + id_dataset] = np.array(r[-1])
                    #     moji += 1
    np.savez_compressed("hiragana.npz", ary)

read_hiragana()



# import struct
# from PIL import Image
#
# def read_record_ETL8G(f):
#     s = f.read(512)
#     r = struct.unpack('>2H8sI4B4H2B30x8128s11x', s)
#     iF = Image.frombytes('F', (64, 63), r[3], 'bit', 4)
#     iL = iF.convert('L')
#     return r + (iL,)
#
#
# filename = 'C:/Users/lele.chen/Downloads/ETL8B/ETL8B/ETL8B2C1'
# id_category = 0
# new_img = Image.new('1', (64*16, 64*10))
#
# with open(filename, 'r') as f:
#     f.seek((id_category * 160 + 1) * 512)
#     for i in range(160):
#         r = read_record_ETL8G(f)
#         new_img.paste(r[-1], (64*(i%16), 64*(i/16)))
#
# iI = Image.eval(new_img, lambda x: not x)
# fn = 'ETL8B2_{:03d}.png'.format(id_category)
# iI.save(fn, 'PNG')

# im = Image.fromarray(image)
# dir_name = './data/test/' + '%0.5d' % char_dict[tagcode_unicode]
# if not os.path.exists(dir_name):
#     os.mkdir(dir_name)
# im.convert('RGB').save(dir_name + '/' + tagcode_unicode + str(test_counter) + '.png')
# test_counter += 1
