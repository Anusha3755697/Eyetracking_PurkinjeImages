import matplotlib.pyplot as plt
import csv
import numpy as np
import math
import pandas as pd
 
x=[]
y=[]
a=[]
b=[]
 
with open('C:\\Users\\manju\\OneDrive\\Desktop\\Anusha\\Research project\\Csv_data\\data_Sample_001.csv', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter=',')
    for column in plots:
        x.append(float(column[0]))
        y.append(float(column[1]))
        a.append(float(column[2]))
        b.append(float(column[3]))
        dotProduct = float(column[0])*float(column[2]) + float(column[1])*float(column[3])
         #----------------for three dimensional simply add dotProduct = a*c + b*d  + e*f 
        modOfVector1 = math.sqrt(float(column[0])*float(column[0]) + float(column[1])*float(column[1]))*math.sqrt(float(column[2])*float(column[2]) + float(column[3])*float(column[3])) 
         #----------------- for three dimensional simply add modOfVector = math.sqrt( a*a + b*b + e*e)*math.sqrt(c*c + d*d +f*f) 
        angle = dotProduct/modOfVector1
        #print("Cosθ =",angle)
        angleInDegree = math.degrees(math.acos(angle))
        print("θ =",angleInDegree,"°")
        distance = math.sqrt(((float(column[0])-float(column[2]))**2) + (float(column[1])-float(column[3]))**2)
        print ("Distance =",distance)
        with open('C:\\Users\\manju\\OneDrive\\Desktop\\Anusha\\Research project\\Csv_data\\graph12_06.csv', 'a', newline='') as outfile:   
            fieldnames = ['angle','degree','distance']
            output = csv.DictWriter(outfile, fieldnames=fieldnames)
            output.writeheader()
            output.writerow({'angle': angle , 'degree':angleInDegree , 'distance':distance})
            output.writerow({})
            outfile.close()
#fig, (ax1, ax2) = plt.subplots(2)
#fig.suptitle('plot for pupil and 1st purkinje coordinates')
plt.scatter(x, y)
plt.scatter(a, b)
plt.xlabel('Pupil_X,PIX')
plt.ylabel('Pupil_Y,PIY')
plt.title('Graph of coordinates of Pupil and First Purkinje Image')

#plt.scatter(angle,distance)

plt.show()