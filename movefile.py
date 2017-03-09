import os
import shutil
from PIL import Image

dic = {
"A":"1",
"BA":"2",
"BE":"3",
"BI":"4",
"BO":"5",
"BU":"6",
"CHI":"7",
"DA":"8",
"DE":"9",
"DI":"10",
"DO":"11",
"DU":"12",
"E":"13",
"FU":"14",
"GA":"15",
"GE":"16",
"GI":"17",
"GO":"18",
"GU":"19",
"HA":"20",
"HE":"21",
"HEI":"22",
"HI":"23",
"HO":"24",
"I":"25",
"JI":"26",
"KA":"27",
"KE":"28",
"KI":"29",
"KO":"30",
"KU":"31",
"MA":"32",
"ME":"33",
"MI":"34",
"MO":"35",
"MU":"36",
"N":"37",
"NA":"38",
"NE":"39",
"NI":"40",
"NO":"41",
"NU":"42",
"O":"43",
"PA":"44",
"PE":"45",
"PI":"46",
"PO":"47",
"PU":"48",
"RA":"49",
"RE":"50",
"RI":"51",
"RO":"52",
"RU":"53",
"SA":"54",
"SE":"55",
"SHI":"56",
"SO":"57",
"SU":"58",
"TA":"59",
"TE":"60",
"TO":"61",
"TSU":"62",
"U":"63",
"WA":"64",
"YA":"65",
"YO":"66",
"YU":"67",
"ZA":"68",
"ZE":"69",
"ZO":"70",
"ZU":"71"
}

def get_imagesfile(data_dir):
    filenames = []
    for root, sub_folder, file_list in os.walk(data_dir):
        for file_path in file_list:
            filename = os.path.join(root, file_path)
            if(int(file_path.split("_")[1])>1300):
                _path = os.path.join("C:\\data\\ETL8G\\test",dic[root.replace(".HIRA","").split('\\')[-1]])
                if not os.path.exists(_path):
                    os.makedirs(_path)
                shutil.copyfile(filename,os.path.join(_path, file_path))
            else:
                _path = os.path.join("C:\\data\\ETL8G\\train", dic[root.replace(".HIRA", "").split('\\')[-1]])
                if not os.path.exists(_path):
                    os.makedirs(_path)
                shutil.copyfile(filename, os.path.join(_path, file_path))
get_imagesfile("C:\\data\\ETL8G")