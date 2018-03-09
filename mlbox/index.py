from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *

paths = ["/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_2010.csv",
         "/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_2011.csv",
        "/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_2012.csv",
"/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_2013.csv",
"/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_2014.csv",
"/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_2015.csv",
"/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_2016.csv",
"/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_201701.csv",
"/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_201702.csv",
"/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_201703.csv",
"/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_201704.csv",
"/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_201705.csv",
"/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_201706.csv"] #to modify
target_name = "target"

data = Reader(sep=",").train_test_split(paths, target_name)  #reading
data = Drift_thresholder().fit_transform(data)

Optimiser().evaluate(None, data)