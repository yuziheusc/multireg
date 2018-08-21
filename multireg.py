import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats
import matplotlib.pyplot as plt

def do_regression(data_frame, response_var, predicators):
    Y = np.array(data_frame[response_var])
    X = np.array(data_frame[predicators])
    X = sm.add_constant(X)
    linear_model = sm.OLS(Y,X)
    lr = linear_model.fit()

    ## make a plot
    if(len(predicators) == 1):
        ## plot the data points
        plt.clf()
        plt.scatter(X[:,1], Y[:], s=0.5, c='b')
        ## plot the fitting line
        x1 = np.arange(min(X[:,1]), max(X[:,1]), (max(X[:,1])-min(X[:,1]))*0.01 )
        y1 = x1 * lr.params[1] + lr.params[0]
        plt.plot(x1,y1,'b--')
        plt.xlabel(predicators[0])
        plt.ylabel(response_var)
        plt.title(response_var+"_vs_"+predicators[0])
        fnou = response_var+"_vs_"+predicators[0]+".pdf"
        plt.savefig(fnou)
        #plt.show()
        
    
    print("fitting parameters : ")
    for i in range(len(lr.params)):
        print("  %4d  %6.3g"%(i,lr.params[i]))
    
    print("standard error : ")
    for i in range(len(lr.bse)):
        print("  %4d  %6.3g"%(i,lr.bse[i]))
        
    print("p-value : ")
    for i in range(len(lr.pvalues)):
        print("  %4d  %6.3g"%(i,lr.pvalues[i]))
    
    #print(lr.ssr)
    print("r-square : ", lr.rsquared)

    print("** BreuschPagan test **")
    test = sms.het_breushpagan(lr.resid, lr.model.exog)
    name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
    for i in range(len(test)):
        print("%s : %5.3f"%(name[i],test[i]))

    print("** Normality of residual **")
    k2, p = stats.normaltest(lr.resid)
    print("p-value = ", p)
    
        
if __name__ == "__main__":
    if(len(sys.argv) != 2):
        print("ERROR: Wrong number of args.")
        sys.exit()
    fnin = sys.argv[1]
    print("\nSTART\n")
    print("Script file : %s"%(fnin))
    try:
        fpin = open(fnin)
    except IOError:
        print("ERROR: No script file found.")
        sys.exit()
    with fpin:
        #fp_data = None
        data_frame = None
        for line in fpin:
            line = line.lstrip()
            line = line.strip('\n')
            if(line == ""): continue
            if(line[0] == "#"): continue
            #print("-",line,"-")

            ## data file description
            if(line[0:4]=="FILE"):
                fn_data = line[4:].lstrip(" (").rstrip(" )")
                print("\nOpen data file :", fn_data)
                data_frame = pd.read_csv(fn_data)
                
            ## regression description
            if(line[0:3]=="RES"):
                line = line.split(';')
                
                ## filter the response and predicator list
                for i in range(2):
                    line[i] = line[i][3:].replace('(',' ').replace(')',' ').replace(',',' ')
                    line[i] = line[i].lstrip().rstrip()
                    line[i] = line[i].split(' ')
                
                
                if(len(line[0])!=1):
                    print("Error: One and only one response var is allowed.")
                    sys.exit()
                    
                response_var = line[0][0]
                predicators = line[1]
                #print("Regression on ['%s'] with predicators %s"%(response_var, predicators) )
                print("\nRegression: ")
                print("Response : ['%s']"%(response_var))
                print("Predicators : %s"%(predicators))

                do_regression(data_frame, response_var, predicators)
                
                print("\n")

    print("\nEND.")
