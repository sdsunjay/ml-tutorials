import os

def readFile(filename):
    with open(filename) as f:
        for line in f:
            if(line != 'Symbol,Date,Close,High,Low,Open,Volume\n'):
                os.rename(filename, "new/"+filename)
                print('Moved ' + filename)
            break

for filename in os.listdir(os.getcwd()):
    readFile(filename)
