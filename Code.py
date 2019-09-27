import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

Data = pd.read_excel(io="Training_Data.xlsx", sheet_name="E0_Mod")
D = pd.DataFrame(Data[['HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']].T)
D2 = pd.DataFrame(Data[['HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']].T)

#%% Level (1)

for x in range (11):
    for i in range (255):
        if D.iloc[x][i] >= (max(D.iloc[x][:])-min(D.iloc[x][:]))/2 :
            D.iloc[x][i] = 1
        else:
            D.iloc[x][i] = 0      
DT = D.T

#%% Computing Entropy(S) and Gains

columns = list(DT)
Sammary = pd.DataFrame()
e1, e2, Gain, Exclude = [], [], [], []
t = Data['FTR'].count()
xFTR_v = Data['FTR'].value_counts()
entropy_S = -(xFTR_v[0]/t)*np.log2(xFTR_v[0]/t)-(xFTR_v[1]/t)*np.log2(xFTR_v[1]/t)

zew = []
for x in columns:
    zew.append(DT[x].value_counts())

for x in range(0,12):
    no_1, no_0 = 0, 0
    for i in range(255):
        if DT.iloc[i][x] == 1 and Data.iloc[i]['FTR'] == 'H' :
            no_1 = no_1 + 1
        if DT.iloc[i][x] == 0 and Data.iloc[i]['FTR'] == 'H' :
            no_0 = no_0 + 1

    nox_1 = zew[x][1]-no_1
    nox_0 = zew[x][0]-no_0
    
    entropy_1 = -(no_1/zew[x][1])*np.log2(no_1/zew[x][1])-(nox_1/zew[x][1])*np.log2(nox_1/zew[x][1])
    entropy_0 = -(no_0/zew[x][0])*np.log2(no_0/zew[x][0])-(nox_0/zew[x][0])*np.log2(nox_0/zew[x][0])
    xGain = entropy_S - (zew[x][1]/t)*entropy_1 - (zew[x][0]/t)*entropy_0
    e1.append(entropy_1), e2.append(entropy_0), Gain.append(xGain)

Sammary['Col'], Sammary['e1'], Sammary['e2'], Sammary['Gain'] = columns, e1, e2, Gain
root = Sammary.loc[Sammary['Gain'].idxmax()].values
print('ROOT IS:',root[0])
Exclude.append(root[0])

#%% Level (2)

def eliminatezeros(no_3, no_1, no_0):
    tn = no_3+no_1+no_0
    nox_3 = tn-no_3
    nox_1 = tn-no_1
    nox_0 = tn-no_0
    if no_0 == 0:
        no_0 = tn
        nox_0 = tn
    if no_1 == 0:
        no_1 = tn
        nox_1 = tn
    if no_3 == 0:
        no_3 = tn
        nox_3 = tn
    return no_3, no_1, no_0, nox_3, nox_1, nox_0, tn

for x in range (11):
    for i in range (255):
        if D2.iloc[x][i] >= max(D2.iloc[x][:])-((max(D2.iloc[x][:])-min(D2.iloc[x][:]))/3) :
            D2.iloc[x][i] = 3
        elif D2.iloc[x][i] >= ((max(D2.iloc[x][:])-min(D2.iloc[x][:]))/3)+min(D2.iloc[x][:]) :
            D2.iloc[x][i] = 1
        else:
            D2.iloc[x][i] = 0

entropy_L = root[1]
entropy_R = root[2]
D2T = D2.T
#print('Entropy(R): \nLeft:',Entropy_L,'\nRight',Entropy_R)

#%% Computing Entropy(S) and Gains - Left Side

e1, e2, e3, lGain = [], [], [], []
for x in columns:
    if x not in [ i for i in Exclude]:
        no_3, no_1, no_0 = 0, 0, 0
        for i in range(255):
            if D2T.iloc[i][x] == 3 and DT.iloc[i][Exclude[0]] == 1 :
                no_3 = no_3 + 1
            if D2T.iloc[i][x] == 1 and DT.iloc[i][Exclude[0]] == 1 :
                no_1 = no_1 + 1
            if D2T.iloc[i][x] == 0 and DT.iloc[i][Exclude[0]] == 1 :
                no_0 = no_0 + 1
                    
        no_3, no_1, no_0, nox_3, nox_1, nox_0, tn = eliminatezeros(no_3, no_1, no_0)
        
        entropy_3 = -(no_3/tn)*np.log2(no_3/tn)-(nox_3/tn)*np.log2(nox_3/tn)
        entropy_1 = -(no_1/tn)*np.log2(no_1/tn)-(nox_1/tn)*np.log2(nox_1/tn)
        entropy_0 = -(no_0/tn)*np.log2(no_0/tn)-(nox_0/tn)*np.log2(nox_0/tn)
        xGain = entropy_L - ((no_3/zew[2][1])*entropy_3 - (no_1/zew[2][1])*entropy_1 - (no_0/zew[2][1])*entropy_0)
        e1.append(entropy_3), e2.append(entropy_1), e3.append(entropy_0), lGain.append(xGain)
    else:
        e1.append(0), e2.append(0), e3.append(0), lGain.append(0)
        
Sammary['e1L'], Sammary['e2L'], Sammary['e3L'], Sammary['lGain'] = e1, e2, e3, lGain
LS = Sammary.loc[Sammary['lGain'].idxmax()].values
print('LEFT SIDE IS:', LS[0])
Exclude.append(LS[0])

#%% Computing Entropy(S) and Gains - Right Side

e1, e2, e3, rGain = [], [], [], []
for x in columns:
    if x not in [ i for i in Exclude]:
        no_3, no_1, no_0 = 0, 0, 0
        for i in range(255):
            if D2T.iloc[i][x] == 3 and DT.iloc[i][Exclude[0]] == 0 :
                no_3 = no_3 + 1
            if D2T.iloc[i][x] == 1 and DT.iloc[i][Exclude[0]] == 0 :
                no_1 = no_1 + 1
            if D2T.iloc[i][x] == 0 and DT.iloc[i][Exclude[0]] == 0 :
                no_0 = no_0 + 1
                    
        no_3, no_1, no_0, nox_3, nox_1, nox_0, tn = eliminatezeros(no_3, no_1, no_0)
        
        entropy_3 = -(no_3/tn)*np.log2(no_3/tn)-(nox_3/tn)*np.log2(nox_3/tn)
        entropy_1 = -(no_1/tn)*np.log2(no_1/tn)-(nox_1/tn)*np.log2(nox_1/tn)
        entropy_0 = -(no_0/tn)*np.log2(no_0/tn)-(nox_0/tn)*np.log2(nox_0/tn)
        xGain = entropy_R - ((no_3/zew[2][0])*entropy_3 - (no_1/zew[2][0])*entropy_1 - (no_0/zew[2][0])*entropy_0)
        e1.append(entropy_3), e2.append(entropy_1), e3.append(entropy_0), rGain.append(xGain)
    else:
        e1.append(0), e2.append(0), e3.append(0), rGain.append(0)

Sammary['e1R'], Sammary['e2R'], Sammary['e3R'], Sammary['rGain'] = e1, e2, e3, rGain
RS = Sammary.loc[Sammary['rGain'].idxmax()].values
print('RIGHT SIDE IS:', RS[0])
Exclude.append(RS[0])

#%% level (3)

zewl = D2T[Exclude[1]].value_counts()
entropy_L2L, entropy_L2M, entropy_L2R = LS[6], LS[5], LS[4]
zewr = D2T[Exclude[2]].value_counts()
entropy_R2L, entropy_R2M, entropy_R2R = RS[10], RS[9], RS[8]

#%% Computing Entropy(S) and Gains - Level 2 - Left Side - L

e1, e2, e3, llGain = [], [], [], []
for x in columns:
    if x not in [ i for i in Exclude]:
        no_3, no_1, no_0 = 0, 0, 0
        for i in range(255):
            if D2T.iloc[i][x] == 3 and D2T.iloc[i][Exclude[1]] == 3 :
                no_3 = no_3 + 1
            if D2T.iloc[i][x] == 1 and D2T.iloc[i][Exclude[1]] == 3 :
                no_1 = no_1 + 1
            if D2T.iloc[i][x] == 0 and D2T.iloc[i][Exclude[1]] == 3 :
                no_0 = no_0 + 1
                    
        no_3, no_1, no_0, nox_3, nox_1, nox_0, tn = eliminatezeros(no_3, no_1, no_0)
        
        entropy_3 = -(no_3/tn)*np.log2(no_3/tn)-(nox_3/tn)*np.log2(nox_3/tn)
        entropy_1 = -(no_1/tn)*np.log2(no_1/tn)-(nox_1/tn)*np.log2(nox_1/tn)
        entropy_0 = -(no_0/tn)*np.log2(no_0/tn)-(nox_0/tn)*np.log2(nox_0/tn)
        xGain = entropy_L2L - ((no_3/zewl[3])*entropy_3 - (no_1/zewl[3])*entropy_1 - (no_0/zewl[3])*entropy_0)
        e1.append(entropy_3), e2.append(entropy_1), e3.append(entropy_0), llGain.append(xGain)
    else:
        e1.append(0), e2.append(0), e3.append(0), llGain.append(0)

Sammary['e1ll'], Sammary['e2ll'], Sammary['e3ll'], Sammary['llGain'] = e1, e2, e3, llGain
LL = Sammary.loc[Sammary['llGain'].idxmax()].values
print('LEFT LEFT IS:', LL[0])
Exclude.append(LL[0])

#%% Computing Entropy(S) and Gains - Level 2 - Left Side -M

e1, e2, e3, lmGain = [], [], [], []
for x in columns:
    if x not in [ i for i in Exclude]:
        no_3, no_1, no_0 = 0, 0, 0
        for i in range(255):
            if D2T.iloc[i][x] == 3 and D2T.iloc[i][Exclude[1]] == 1 :
                no_3 = no_3 + 1
            if D2T.iloc[i][x] == 1 and D2T.iloc[i][Exclude[1]] == 1 :
                no_1 = no_1 + 1
            if D2T.iloc[i][x] == 0 and D2T.iloc[i][Exclude[1]] == 1 :
                no_0 = no_0 + 1

        no_3, no_1, no_0, nox_3, nox_1, nox_0, tn = eliminatezeros(no_3, no_1, no_0)
        
        entropy_3 = -(no_3/tn)*np.log2(no_3/tn)-(nox_3/tn)*np.log2(nox_3/tn)
        entropy_1 = -(no_1/tn)*np.log2(no_1/tn)-(nox_1/tn)*np.log2(nox_1/tn)
        entropy_0 = -(no_0/tn)*np.log2(no_0/tn)-(nox_0/tn)*np.log2(nox_0/tn)
        xGain = entropy_L2M - ((no_3/zewl[1])*entropy_3 - (no_1/zewl[1])*entropy_1 - (no_0/zewl[1])*entropy_0)
        e1.append(entropy_3), e2.append(entropy_1), e3.append(entropy_0), lmGain.append(xGain)
    else:
        e1.append(0), e2.append(0), e3.append(0), lmGain.append(0)
        
Sammary['e1lm'], Sammary['e2lm'], Sammary['e3lm'], Sammary['lmGain'] = e1, e2, e3, lmGain
LM = Sammary.loc[Sammary['lmGain'].idxmax()].values
print('LEFT MIDDLE IS:', LM[0])
Exclude.append(LM[0])

#%% Computing Entropy(S) and Gains - Level 2 - Left Side (R)

e1, e2, e3, lrGain = [], [], [], []
for x in columns:
    if x not in [ i for i in Exclude]:
        no_3, no_1, no_0 = 0, 0, 0
        for i in range(255):
            if D2T.iloc[i][x] == 3 and D2T.iloc[i][Exclude[1]] == 0 :
                no_3 = no_3 + 1
            if D2T.iloc[i][x] == 1 and D2T.iloc[i][Exclude[1]] == 0 :
                no_1 = no_1 + 1
            if D2T.iloc[i][x] == 0 and D2T.iloc[i][Exclude[1]] == 0 :
                no_0 = no_0 + 1

        no_3, no_1, no_0, nox_3, nox_1, nox_0, tn = eliminatezeros(no_3, no_1, no_0)
        
        entropy_3 = -(no_3/tn)*np.log2(no_3/tn)-(nox_3/tn)*np.log2(nox_3/tn)
        entropy_1 = -(no_1/tn)*np.log2(no_1/tn)-(nox_1/tn)*np.log2(nox_1/tn)
        entropy_0 = -(no_0/tn)*np.log2(no_0/tn)-(nox_0/tn)*np.log2(nox_0/tn)
        xGain = entropy_L2R - ((no_3/zewl[0])*entropy_3 - (no_1/zewl[0])*entropy_1 - (no_0/zewl[0])*entropy_0)
        e1.append(entropy_3), e2.append(entropy_1), e3.append(entropy_0), lrGain.append(xGain)
    else:
        e1.append(0), e2.append(0), e3.append(0), lrGain.append(0)
        
Sammary['e1lr'], Sammary['e2lr'], Sammary['e3lr'], Sammary['lrGain'] = e1, e2, e3, lrGain
LR = Sammary.loc[Sammary['lrGain'].idxmax()].values
print('LEFT RIGHT IS:', LR[0])
Exclude.append(LR[0])



#%% Computing Entropy(S) and Gains - Level 2 - Right Side - L

e1, e2, e3, rlGain = [], [], [], []
for x in columns:
    if x not in [ i for i in Exclude]:
        no_3, no_1, no_0 = 0, 0, 0
        for i in range(255):
            if D2T.iloc[i][x] == 3 and D2T.iloc[i][Exclude[2]] == 3 :
                no_3 = no_3 + 1
            if D2T.iloc[i][x] == 1 and D2T.iloc[i][Exclude[2]] == 3 :
                no_1 = no_1 + 1
            if D2T.iloc[i][x] == 0 and D2T.iloc[i][Exclude[2]] == 3 :
                no_0 = no_0 + 1
                    
        no_3, no_1, no_0, nox_3, nox_1, nox_0, tn = eliminatezeros(no_3, no_1, no_0)
        
        entropy_3 = -(no_3/tn)*np.log2(no_3/tn)-(nox_3/tn)*np.log2(nox_3/tn)
        entropy_1 = -(no_1/tn)*np.log2(no_1/tn)-(nox_1/tn)*np.log2(nox_1/tn)
        entropy_0 = -(no_0/tn)*np.log2(no_0/tn)-(nox_0/tn)*np.log2(nox_0/tn)
        xGain = entropy_R2L - ((no_3/zewr[3])*entropy_3 - (no_1/zewr[3])*entropy_1 - (no_0/zewr[3])*entropy_0)
        e1.append(entropy_3), e2.append(entropy_1), e3.append(entropy_0), rlGain.append(xGain)
    else:
        e1.append(0), e2.append(0), e3.append(0), rlGain.append(0)

Sammary['e1rl'], Sammary['e2rl'], Sammary['e3rl'], Sammary['rlGain'] = e1, e2, e3, rlGain
RL = Sammary.loc[Sammary['rlGain'].idxmax()].values
print('RIGHT LEFT IS:', RL[0])
Exclude.append(RL[0])

#%% Computing Entropy(S) and Gains - Level 2 - Right Side -M

e1, e2, e3, rmGain = [], [], [], []
for x in columns:
    if x not in [ i for i in Exclude]:
        no_3, no_1, no_0 = 0, 0, 0
        for i in range(255):
            if D2T.iloc[i][x] == 3 and D2T.iloc[i][Exclude[2]] == 1 :
                no_3 = no_3 + 1
            if D2T.iloc[i][x] == 1 and D2T.iloc[i][Exclude[2]] == 1 :
                no_1 = no_1 + 1
            if D2T.iloc[i][x] == 0 and D2T.iloc[i][Exclude[2]] == 1 :
                no_0 = no_0 + 1

        no_3, no_1, no_0, nox_3, nox_1, nox_0, tn = eliminatezeros(no_3, no_1, no_0)
        
        entropy_3 = -(no_3/tn)*np.log2(no_3/tn)-(nox_3/tn)*np.log2(nox_3/tn)
        entropy_1 = -(no_1/tn)*np.log2(no_1/tn)-(nox_1/tn)*np.log2(nox_1/tn)
        entropy_0 = -(no_0/tn)*np.log2(no_0/tn)-(nox_0/tn)*np.log2(nox_0/tn)
        xGain = entropy_R2M - ((no_3/zewr[1])*entropy_3 - (no_1/zewr[1])*entropy_1 - (no_0/zewr[1])*entropy_0)
        e1.append(entropy_3), e2.append(entropy_1), e3.append(entropy_0), rmGain.append(xGain)
    else:
        e1.append(0), e2.append(0), e3.append(0), rmGain.append(0)
        
Sammary['e1rm'], Sammary['e2rm'], Sammary['e3rm'], Sammary['rmGain'] = e1, e2, e3, rmGain
RM = Sammary.loc[Sammary['rmGain'].idxmax()].values
print('RIGHT MIDDLE IS:', RM[0])
Exclude.append(RM[0])

#%% Computing Entropy(S) and Gains - Level 2 - Right Side (R)

e1, e2, e3, rrGain = [], [], [], []
for x in columns:
    if x not in [ i for i in Exclude]:
        no_3, no_1, no_0 = 0, 0, 0
        for i in range(255):
            if D2T.iloc[i][x] == 3 and D2T.iloc[i][Exclude[2]] == 0 :
                no_3 = no_3 + 1
            if D2T.iloc[i][x] == 1 and D2T.iloc[i][Exclude[2]] == 0 :
                no_1 = no_1 + 1
            if D2T.iloc[i][x] == 0 and D2T.iloc[i][Exclude[2]] == 0 :
                no_0 = no_0 + 1

        no_3, no_1, no_0, nox_3, nox_1, nox_0, tn = eliminatezeros(no_3, no_1, no_0)

        entropy_3 = -(no_3/tn)*np.log2(no_3/tn)-(nox_3/tn)*np.log2(nox_3/tn)
        entropy_1 = -(no_1/tn)*np.log2(no_1/tn)-(nox_1/tn)*np.log2(nox_1/tn)
        entropy_0 = -(no_0/tn)*np.log2(no_0/tn)-(nox_0/tn)*np.log2(nox_0/tn)
        xGain = entropy_R2R - ((no_3/zewr[0])*entropy_3 - (no_1/zewr[0])*entropy_1 - (no_0/zewr[0])*entropy_0)
        e1.append(entropy_3), e2.append(entropy_1), e3.append(entropy_0), rrGain.append(xGain)
    else:
        e1.append(0), e2.append(0), e3.append(0), rrGain.append(0)
        
Sammary['e1rr'], Sammary['e2rr'], Sammary['e3rr'], Sammary['rrGain'] = e1, e2, e3, rrGain
RR = Sammary.loc[Sammary['rrGain'].idxmax()].values
print('RIGHT RIGHT IS:', RR[0])
Exclude.append(RR[0])

#%%













