from flask import Flask, request, redirect, render_template
import io
import base64
import numpy as np
import pandas as pd
from flask import Response
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


nActiveCasesPeak = 4800000
nCurrentActiveCases = 2813658
presentNewCases = 350000
PeakNewCases = 460000
nCurrentNewActiveCases = 130900
durationToPeak =  20 #days
RecoveryRatio = nCurrentNewActiveCases/presentNewCases


VolOxyReqPerson = 550
rOxSup = 0.04
rLtoTn = 0.00114
PolyRegCases = 4
PolyRegOxy = 3
# input variables (dynamic input)
# indic = (input("Would you like to enter your own data? Type Y/N: "))
# if indic == 'Y':
#     VolOxyReqPerson =  float(input("Enter Daily Anticipated Oxygen needed per person in liter (Human Avg is 500 to 550 liters): "))
#     rOxSup = float(input("Enter proportion of patients that would need oxygen  support (typically 0.03 to 0.04): "))
#     PolyRegCases = int(input("What degree of polynomial would you like to regress the new and total active cases prediction to? Please enter an integer (recommended:  4): " ))
#     PolyRegOxy = int(input("What degree of polynomial would you like to regress the oxygen supply prediction to? Please enter an integer (recommended:  3): " ))
#     # dynamic code based input of statewise data not included yet but can later be included for potential future use

# if indic == 'N':
#     print("\nScript will proceed with default values gathered on 27th of April, 2021 for India from online sources\n")

# else:
#     print("\nInvalid Input (user did not type Y or N). Script will proceed with default values gathered on 27th of April, 2021 for India from online sources\n")

TotalActiveCaseList = [0]*durationToPeak
TotalActiveCaseList[0] = nCurrentActiveCases
TotalActiveCaseList[19] = nActiveCasesPeak

DailyAcceleration =  ((PeakNewCases - presentNewCases)/10)
IndexList = list(range(1,21))
IndexList1 = [0]*10
IndexList1[0:9] = list(range(1,11))
IndexList1[10] = 20


newCaselist = [0] * 20
newCaselist[0] = presentNewCases

for i in range(1,10):
    newCaselist[i] = (int)(newCaselist[i-1] + DailyAcceleration)

for j in range(1,10):
    TotalActiveCaseList[j] = (int)(TotalActiveCaseList[j-1] + RecoveryRatio*newCaselist[j])


RegressList = [0]*11
RegressList[0:10] = TotalActiveCaseList[0:10]
RegressList[10] = TotalActiveCaseList[19]


PolyModel = np.poly1d(np.polyfit(IndexList1,RegressList, PolyRegCases))
PolyLine = np.linspace(1, 20, nActiveCasesPeak)

r2 = r2_score(RegressList, PolyModel(IndexList1))


for k in range(10,20):
    TotalActiveCaseList[k] = int(PolyModel(k+1))
    
for a in range(10,20):    
    newCaselist[a] = (int)((TotalActiveCaseList[a] - TotalActiveCaseList[a-1])/RecoveryRatio)
    if newCaselist[a] <= 0:
        newCaselist[a] = newCaselist[a-1]


massOxSupNeed = [0]*20
for p in range(0,20):
    nDailyOxSupNeed = int(rOxSup * newCaselist[p])
    massOxSupNeed[p] = nDailyOxSupNeed * VolOxyReqPerson * rLtoTn


x = [2, 4, 6, 8, 10]
y = [2800000 ,3100000, 3300000, 3500000, 3700000]


#print("\nR Squared Value for cubic polynomial best fit curve for Oxygen Demand in India April 26th through May 16th:")
#print("{:.6f}".format(r2))
RealModel = np.poly1d(np.polyfit(x,y, 2))
RealLine = np.linspace(1, 20, 100)

OxyModel = np.poly1d(np.polyfit(range(1,18),massOxSupNeed[0:17], PolyRegOxy))
OxyLine = np.linspace(1, 17, 100)

OxyR2 = r2_score(massOxSupNeed[0:17], OxyModel(range(1,18)))

#print("R Squared Value for quartic polynomial regression of Total Active Cases in India April 26th through May 16th:")
#print("{:.6f}".format(OxyR2))


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/")
def home():
    base = request.base_url
    return render_template("homepage.html")
    #return  """<xmp>
     #          Welcome to the Homepage of the OxygenDemandPredictor of the CovidUpchaar App

      #         To view the  predictive plot of Active Covid-19-cases please type /activecases after the homepage link in the url bar.
              
       #        To view the  predictive plot of Oxygen Demand till the peak in May, please type /oxydemand after the homepage link in the url bar.</xmp>"""

@app.route("/activecases")
def activeCases():
    img = Figure()
    ax = img.subplots()
    ax.scatter(IndexList1,RegressList)
    ax.scatter(x, y,color = 'red' )
    ax.plot(RealLine, RealModel(RealLine), color = 'red', label = 'Real Data Based Line') 
    ax.plot(PolyLine, PolyModel(PolyLine), label = 'Predicted Data Based Line')
    ax.legend()
    ax.set_title('Predictive Polynomial Regression of Active Covid Cases in India', pad =20)
    ax.set_xlabel('Days from April 26th, 2021')
    ax.set_ylabel('Projected Total Active Covid-19 Cases in India')
    #ax.xticks(np.arange(min(IndexList1), max(IndexList1)+1, 1))
    tempBuf = io.BytesIO()
    img.savefig(tempBuf, format="png")
    data = base64.b64encode(tempBuf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"


@app.route("/oxydemand")
def OxyDemand():
    img = Figure()
    ax = img.subplots()

    ax.scatter(range(1,18), massOxSupNeed[0:17])
    ax.plot(OxyLine,OxyModel(OxyLine))
    ax.set_title('Predictive Polynomial Best Fit Curve for Oxygen Demand in India till Peak in May, 2021', pad = 20)
    ax.set_xlabel('Days from April 26th, 2021')
    ax.set_ylabel('Oxygen Demand in India(Tonnes)')
    #ax.xticks(np.arange(1, 18, 1))
    #ax.xticks(np.arange(min(IndexList1), max(IndexList1)+1, 1))
    tempBuf = io.BytesIO()
    img.savefig(tempBuf, format="png")
    data = base64.b64encode(tempBuf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"
    
    
