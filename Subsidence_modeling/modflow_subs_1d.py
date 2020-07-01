import numpy as np
import flopy

# Imports
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf

plt.close('all')

# this code models land subsidence with a known hydraulic head history. 
# Current head history data is from the Central Valley in California.
# You can load in your own head data and replace hinterp (head) and t3 (time).
# current units for length and time are meters and days, respectively.
# Time step is monthly, so head data are given for each month.
# To run this you need to install the flopy python package


# References on subsidence: 
# summary of subsidence in the Central Valley: https://pubs.usgs.gov/pp/0437h/report.pdf
# overview of subsidence and documentation on modeling subsidence with MODFLOW: https://pubs.usgs.gov/of/2003/ofr03-233/pdf/ofr03233.pdf
# Ask Ryan Smith for a copy of Riley's 1998 paper, which has a great summary of subsidence
# Subsidence and InSAR: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016WR019861
# Example of modeling subsidence: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018WR024185

# you can modify these to match the observed data
# See Sneed, 2001 (https://pubs.usgs.gov/of/2001/ofr01-35/pdf/ofr0135.pdf) for representative values
Kv=1E-8 # meters per day
nclay=20 # number of clay layers
claythick=10 # m thickness of a single clay layer
ss = 1.E-5 # m^-1 specific storage (usually the same as sske in areas with lots of compaction)
sske=1.E-5 # m^-1 elastic skeletal specific storage
sskv=1.E-3 # m^-1 inelastic skeletal specific storage

# 1 for Tulare region (recent subsidence), 2 for Los Banos/Kettleman City area (historic subsidence)


subs_area=2

if subs_area==2:
    dat=np.genfromtxt('Elastic_1.csv',delimiter=',')
    head=dat[3::,1]*.3048
    time_bound=dat[3::,0]
    t2=time_bound+2.5/12 # march 15
    t3=np.linspace(t2[0],t2[-1],12*len(t2))
    hinterp=np.interp(t3,t2,head)+5*np.cos(3*np.pi*(t3-2.5/12))
    
#    test=np.linspace(head[0]+30,head[0],30)
#    test2=np.linspace(time_bound[0]-30,time_bound[0]-1,30)
    
#    head=np.concatenate((test,head))
#    time_bound=np.concatenate((test2,time_bound))

if subs_area==1:
    dat=np.genfromtxt('Inelastic_2.csv',delimiter=',')
    head=dat[:,1]*.3048
    time_bound=dat[:,0]
    t2=time_bound+2.5/12 # march 15
    t3=np.linspace(t2[0],t2[-1],12*len(t2))
    hinterp=np.interp(t3,t2,head)+1*np.cos(3*np.pi*(t3-2.5/12))

units_per_day=1
period_length=365/12
nstp=10
Lx = 10. #meters
Ly = 10. #meters
ztop = 20.
zbot = -20.
nlay = 1
nrow = 3
ncol = 3

# Model domain and grid definition

delr = Lx / ncol
delc = Ly / nrow
delv = (ztop - zbot) / nlay
botm = np.linspace(ztop, zbot, nlay + 1)
hk = 1000/units_per_day # meters/day
vka = 0.004/units_per_day # meters/hour
laytyp = 0
preconsolidation_head=hinterp[0]

subs_format=[0, 41, 0, 42, 0, 42, 0, 41, 0, 43, 0, 43]
subs_save=[0, 1000, 0, 1000, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Variables for the BAS package
# Note that changes from the previous tutorial!
ibound = np.ones((nlay, nrow, ncol), dtype=np.int32)
strt = head[0] * np.ones((nlay, nrow, ncol), dtype=np.float32)

# Time step parameters
nper = len(hinterp)
perlen = period_length
#perlen = [1, period_length, period_length, period_length, period_length, period_length]
#nstp = [1, nstps, nstps, nstps, nstps, nstps]
#steady = [True, False, False, False, False, False]

# Flopy objects
modelname = 'tutorial2'
mf = flopy.modflow.Modflow(modelname, exe_name='mf2005')
dis = flopy.modflow.ModflowDis(mf, nlay, nrow, ncol, delr=delr, delc=delc,
                               top=ztop, botm=botm[1:],
                               nper=nper, perlen=perlen, nstp=nstp, steady=False)
bas = flopy.modflow.ModflowBas(mf, ibound=ibound, strt=strt,stoper=2)
lpf = flopy.modflow.ModflowLpf(mf, hk=hk, vka=vka, ss=ss, laytyp=laytyp, ipakcb=53)
#    pcg = flopy.modflow.ModflowPcg(mf,rclose=.1,hclose=.01)
sip = flopy.modflow.mfsip.ModflowSip(mf)
sub = flopy.modflow.ModflowSub(mf,dhc=preconsolidation_head,hc=preconsolidation_head,dstart=head[0],nndb=0,isuboc=1,
                               dp=np.tile([Kv,sske,sskv],(1,1)),dz=claythick,rnb=nclay,
                               ids15=subs_format,
                               ids16=subs_save)

# Make list for stress periods
stress_period_data0={}
for tt in range(len(hinterp)):
    bound_sp = []
    for il in range(nlay):
        condleft = 10000
        condright = 10000
        for ir in range(nrow):
            bound_sp.append([il, ir, 0, hinterp[tt], condleft])
            bound_sp.append([il, ir, ncol - 1, hinterp[tt], condright])
    stress_period_data0.update({tt:bound_sp})
print('Adding GHBs for stress periods')

# Create the flopy ghb object
ghb = flopy.modflow.ModflowGhb(mf, stress_period_data=stress_period_data0)

#lrcd = {0:boundary}   #this chd will be applied to all
#chd = flopy.modflow.ModflowChd(mf, stress_period_data=lrcd)

# Create the well package
# Remember to use zero-based layer, row, column indices!

stress_period_data = {}
for kper in range(nper):
    stress_period_data[(kper, 1)] = ['save head',
                                        'save drawdown',
                                        'save budget',
                                        'print head',
                                        'print budget']
    stress_period_data[(kper, 99)] = ['save head',
                                        'save drawdown',
                                        'save budget',
                                        'print head',
                                        'print budget']
oc = flopy.modflow.ModflowOc(mf, stress_period_data=stress_period_data,
                             compact=True)

# Write the model input files
mf.write_input()

# Run the model
success, mfoutput = mf.run_model(silent=True, pause=False, report=True)
#if not success:
#    raise Exception('MODFLOW did not terminate normally.')
    
# Create the headfile and budget file objects
headobj = bf.HeadFile(modelname + '.hds')
#subsobj = bf.binaryread(modelname + '.cbc')
times = headobj.get_times()
#cbb = bf.CellBudgetFile(modelname + '.cbc')

sobj = flopy.utils.HeadFile(modelname + '.subsidence.hds', 
                            text='SUBSIDENCE')

idx = (0, int(nrow / 2) , int(ncol / 2) )
ts = headobj.get_ts(idx)
ts_subs=sobj.get_ts(idx)

plt.figure()
plt.subplot(2, 1, 1)
plt.ylabel('head, m')
plt.plot(ts[:,0]/365+time_bound[0],ts[:,1], 'b-')
plt.subplot(2,1,2)
plt.plot(ts_subs[:,0]/365+time_bound[0],ts_subs[:,1],'r--')
plt.ylabel('subsidence, m')
