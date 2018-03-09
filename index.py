from forex import Forex
import sys

print(sys.argv[1])
if __name__ == "__main__":
    forex = Forex("./training/{}/".format(sys.argv[1]))
    forex.run()

    #forex.testInitial()
    #forex.test()