import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

def do_regression(data_frame, response_var, predicators):
    Y = np.array(data_frame[response_var])
    X = np.array(data_frame[predicators])
    X = sm.add_constant(X)
    linear_model = sm.OLS(Y,X)
    lr = linear_model.fit()
    #print(X)
    print(lr.params)
    print(lr.bse)
    print(abs(lr.params[1])<1.96*lr.bse[1])
    print(lr.pvalues)
    print(lr.ssr)
    print(lr.rsquared)
    
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
