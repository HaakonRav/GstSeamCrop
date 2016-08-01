
import numpy as np
import matplotlib.pyplot as plt

retargeting_results         = []
first_retargeting_results   = []
initial_latency             = []
transitional_results        = []
recv_send_latency           = []
recv_add_latency            = []
qadd_send_latency           = []
qadd_send_sum               = []
reinit_latency              = []
avg_energy_latency          = []
cpu_results                 = []
mem_results                 = []

# Initial retargeting latency
initial_ret_latency         = []
initial_tot_latency         = []
initial_plu_latency         = []
initial_ret_latency.append(dict())
initial_tot_latency.append(dict())
initial_plu_latency.append(dict())

resolutions = ['640x360','854x480','1280x720']

parse_dir = "parsed_data/"
top_dir = "parsed_data/top/"

#for scenario in range(2,5):
for scenario in range(2,5):
    retargeting_results.append(dict())
    first_retargeting_results.append(dict())
    initial_latency.append(dict())
    transitional_results.append(dict())
    recv_send_latency.append(dict())
    recv_add_latency.append(dict())
    qadd_send_latency.append(dict())
    qadd_send_sum.append(dict())
    avg_energy_latency.append(dict())
    reinit_latency.append(dict())
    cpu_results.append([])
    mem_results.append([])

    file_count = 0


    if scenario < 4:
        ret_dict    = retargeting_results[scenario-2]
        fret_dict   = first_retargeting_results[scenario-2]
        init_dict   = initial_latency[scenario-2]
        trans_dict  = transitional_results[scenario-2]
        rese_dict   = recv_send_latency[scenario-2]
        read_dict   = recv_add_latency[scenario-2]
        qase_dict   = qadd_send_latency[scenario-2]
        sumQ_dict   = qadd_send_sum[scenario-2]
        avge_dict   = avg_energy_latency[scenario-2]
        rein_dict   = reinit_latency[scenario-2]

    if scenario == 4:
        rini_dict = initial_ret_latency[0]
        toni_dict = initial_tot_latency[0]
        plni_dict = initial_plu_latency[0]

    for retarget in range(1,4):
        for framewinsize in range(1,4):
            cpu_name = top_dir +"CPU_" +str(scenario) + "_" + str(retarget) + "_" + str(framewinsize) + ".txt"
            mem_name = top_dir +"MEM_" +str(scenario) + "_" + str(retarget) + "_" + str(framewinsize) + ".txt"
            filename = parse_dir + str(scenario) + "_" + str(retarget) + "_" + str(framewinsize) + "_Parsed.txt"
            input_file = open(filename,'r')

            # Divide the line into the individual measurements.
            for line in input_file:
                splitline = line.split(' ')

                # Typo in the original files.
                if splitline[0] == '858x480':
                    splitline[0] = '854x480'

                if scenario < 4:
                    
                    # Retargeting average
                    if splitline[0] in ret_dict:
                        ret_dict[splitline[0]].append(float(splitline[3]))
                    else:
                        ret_dict[splitline[0]] = [float(splitline[3])]
                    # First retargeting average
                    if splitline[0] in fret_dict:
                        fret_dict[splitline[0]].append(float(splitline[4]))
                    else:
                        fret_dict[splitline[0]] = [float(splitline[4])]
                    # Transitional smoothing time
                    if splitline[0] in trans_dict:
                        trans_dict[splitline[0]].append(float(splitline[5]))
                    else:
                        trans_dict[splitline[0]] = [float(splitline[5])]
                    # Initial latency
                    if splitline[0] in init_dict:
                        init_dict[splitline[0]].append(int(splitline[6]))
                    else:
                        init_dict[splitline[0]] = [int(splitline[6])]
                    # Average recv/send time
                    if splitline[0] in rese_dict:
                        rese_dict[splitline[0]].append(float(splitline[7]))
                    else:
                        rese_dict[splitline[0]] = [float(splitline[7])]
                    # Average recv/add time
                    if splitline[0] in read_dict:
                        read_dict[splitline[0]].append(float(splitline[8]))
                    else:
                        read_dict[splitline[0]] = [float(splitline[8])]
                    # Average qadd/send time
                    if splitline[0] in qase_dict:
                        qase_dict[splitline[0]].append(float(splitline[9]))
                    else:
                        qase_dict[splitline[0]] = [float(splitline[9])]
                    # Sum qadd/send time
                    if splitline[0] in sumQ_dict:
                        sumQ_dict[splitline[0]].append(int(splitline[11]))
                    else:
                        sumQ_dict[splitline[0]] = [int(splitline[11])]
                    # Reinitialization time
                    if float(splitline[13]) != 0:
                        if splitline[0] in rein_dict:
                            rein_dict[splitline[0]].append(float(splitline[13]))
                        else:
                            rein_dict[splitline[0]] = [float(splitline[13])]
                    # Average energy pass latency
                    if splitline[0] in avge_dict:
                        avge_dict[splitline[0]].append(float(splitline[14]))
                    else:
                        avge_dict[splitline[0]] = [float(splitline[14])]
                else:
                    # Average initial total time
                    if splitline[0] in toni_dict:
                        toni_dict[splitline[0]].append(float(splitline[1]))
                    else:
                        toni_dict[splitline[0]] = [float(splitline[1])]
                    # Average initial retargeting time
                    if splitline[0] in rini_dict:
                        rini_dict[splitline[0]].append(float(splitline[2]))
                    else:
                        rini_dict[splitline[0]] = [float(splitline[2])]
                    # Average plugin impact time
                    if splitline[0] in plni_dict:
                        plni_dict[splitline[0]].append(float(splitline[3]))
                    else:
                        plni_dict[splitline[0]] = [float(splitline[3])]

            if scenario < 4:
                cpu_results[scenario-2].append([])
                mem_results[scenario-2].append([])

                cpu = cpu_results[scenario-2][file_count]
                mem = mem_results[scenario-2][file_count]
                cpu_file = open(cpu_name,'r')
                mem_file = open(mem_name,'r')
                for line in cpu_file:
                    cpu.append(float(line.split('\t')[1].replace(',','.').strip()))
                for line in mem_file:
                    mem.append(float(line.split('\t')[1].replace(',','.').strip()))

            file_count += 1

## Pipeline CPU usage without seamcrop.
pipeline_cpu = []
pipeline_avg = [0.0,0.0,0.0]
for i in range(0,3):
    pipeline_cpu.append([])
    for line in open("parsed_data/top/CPU_5_1_"+str(i+1)+".txt",'r'):
        pipeline_cpu[i].append(float(line.split('\t')[1].replace(',','.').strip()))
    pipeline_avg[i] = sum(pipeline_cpu[i]) / len(pipeline_cpu[i])

#print pipeline_cpu
print pipeline_avg
print sum(pipeline_avg) / 3

cpu_avg = []
mem_avg = []
for scenario in range(0,2):
    cpu_avg.append([])
    mem_avg.append([])
    for i in range (0,len(cpu_results[scenario])):
        cpu_avg[scenario].append([])
        cpu_avg[scenario][i] = 0
        mem_avg[scenario].append([])
        mem_avg[scenario][i] = 0
        for item in cpu_results[scenario][i][10:260]:
            cpu_avg[scenario][i] += item
        for item in mem_results[scenario][i][10:260]:
            mem_avg[scenario][i] += item
        cpu_avg[scenario][i] = cpu_avg[scenario][i] / 250.0
        mem_avg[scenario][i] = mem_avg[scenario][i] / 250.0


#for item in cpu_avg:
#    print item

#for item in mem_avg:
#    print item
# Cpu_results = [Scenario][Filenum] up to 9 / 27.
#print avge_dict
#print "TONI:"
#for item in toni_dict:
#    print item

#print mem_results[0][0]
#print cpu_results

minTotalLatency = toni_dict['640x360']
SDTotalLatency = toni_dict['854x480']
HDTotalLatency = toni_dict['1280x720']
minRetLatency = rini_dict['640x360']
SDRetLatency = rini_dict['854x480']
HDRetLatency = rini_dict['1280x720']
minPassOneLatency = avg_energy_latency[0]['640x360']
SDPassOneLatency =  avg_energy_latency[0]['854x480']
HDPassOneLatency = avg_energy_latency[0]['1280x720']

#plt.plot(cpu_results[0][0][10:80])
#plt.plot(cpu_results[0][1][10:80])
#plt.plot(cpu_results[0][2][10:80])
#plt.ylabel("CPU usage(%)")
#plt.xlabel("Seconds (s)")
#plt.axis([0,240,300,650])
#plt.axis([0,70,400,600])
#plt.show()

#plt.show()
#for i in range(mem_results[0][0][10:260]

############################################################################
#### Parse CPU results #####################################################
############################################################################

cpu_frame_window_conf = []

window_count = 0
for scenario in range(0,1):
    cpu_frame_window_conf.append([])
    for measure in cpu_results[scenario]:
        cpu_frame_window_conf[scenario].append([])
        cpu_frame_window_conf[scenario][window_count] = measure[10:80]
        cpu_frame_window_conf[scenario].append([])
        cpu_frame_window_conf[scenario][window_count+1] = measure[100:170]
        cpu_frame_window_conf[scenario].append([])
        cpu_frame_window_conf[scenario][window_count+2] = measure[200:260]
        window_count += 3

retargetfactors = ['0.85','0.75','0.50']

retfac_cpu_usage = [dict(),dict(),dict()]

retfac_cpu_usage[0]['640x360'] = dict()
retfac_cpu_usage[0]['854x480'] = dict()
retfac_cpu_usage[0]['1280x720'] = dict()
retfac_cpu_usage[1]['640x360'] = dict()
retfac_cpu_usage[1]['854x480'] = dict()
retfac_cpu_usage[1]['1280x720'] = dict()
retfac_cpu_usage[2]['640x360'] = dict()
retfac_cpu_usage[2]['854x480'] = dict()
retfac_cpu_usage[2]['1280x720'] = dict()

avg_cpu_usage = [dict(),dict(),dict()]

avg_cpu_usage[0]['640x360'] = 0.0
avg_cpu_usage[0]['854x480'] = 0.0
avg_cpu_usage[0]['1280x720'] = 0.0
avg_cpu_usage[1]['640x360'] = 0.0
avg_cpu_usage[1]['854x480'] = 0.0
avg_cpu_usage[1]['1280x720'] = 0.0
avg_cpu_usage[2]['640x360'] = 0.0
avg_cpu_usage[2]['854x480'] = 0.0
avg_cpu_usage[2]['1280x720'] = 0.0

sum_count = 0.0
for j in range(0,3):
    for n in range(0,3):
        for i in range(0,27,9):
            retfac_cpu_usage[j][resolutions[n]][retargetfactors[i/9]] = \
                    sum(cpu_frame_window_conf[0][((j*3)+n+i)]) / len(cpu_frame_window_conf[0][((j*3)+n+i)])
            avg_cpu_usage[j][resolutions[n]] += sum(cpu_frame_window_conf[0][(j*3)+i+n])
            sum_count+= len(cpu_frame_window_conf[0][(j*3)+i+n])
        avg_cpu_usage[j][resolutions[n]] = avg_cpu_usage[j][resolutions[n]] / sum_count
        sum_count = 0.0

#for i in range(0,3):
#    for item in retfac_cpu_usage[i]:
#        print retfac_cpu_usage[i][item]

minAvgs = []
medAvgs = []
bigAvgs = []

N = 9
ind = np.arange(N)
width = 0.20

avgs = []

summed = 0.0
total_measures = 0
for i in range(0,3):
    avgs.append([])
    for j in range(0,3):
        for n in range(0,3):
            avgs[i].append(retfac_cpu_usage[n][resolutions[i]][retargetfactors[j]])
        summed += sum(avgs[i])
        total_measures += len(avgs[i])
#print avgs

## Prints the average CPU load of all experiments.
#print summed / float(total_measures)

#fig, ax = plt.subplots()
#plt.plot(avgs[0],linestyle='-',marker='^',color='#cde6ff', label='640x360')
#plt.plot(avgs[1], linestyle='-',marker='o',color='#cdffd4', label='854x480')
#plt.plot(avgs[2], linestyle='-',marker='D',color='#cdffd4', label='1280x720')
#bigBars = ax.bar(ind+(width*2),avgs[2], linestyle='-',marker='^',color='#ffcdd2', label='640x360')
#medBars = ax.bar(ind+width,avgs[1], linestyle='-',marker='^',color='#cdffd4', label='854x480')
#bigBars = ax.bar(ind+(width*2),avgs[2], linestyle='-',marker='^',color='#ffcdd2', label='1280x720')

plt.plot(avgs[0],linestyle='-',marker='s',color='0.50', label='640x360')
plt.plot(avgs[1], linestyle='-',marker='o',color='0.40', label='854x480')
plt.plot(avgs[2], linestyle='-',marker='D',color='0.30', label='1280x720')

plt.ylabel("CPU Usage (%)")
plt.xlabel("Configuration")
plt.xticks(ind,[1,2,3,4,5,6,7,8,9])
plt.legend()

ax=plt.gca()
cur_ylim=ax.get_ylim()
ax.set_ylim([495,550])

plt.savefig('../figures/CPU_distribution.png', bbox_inches='tight')
plt.gcf().clear()

#avg_cpu_usage = [dict(),dict(),dict()]

#avg_cpu_usage[0]['640x360'] = 0.0
#avg_cpu_usage[0]['854x480'] = 0.0
#avg_cpu_usage[0]['1280x720'] = 0.0
#avg_cpu_usage[1]['640x360'] = 0.0
#avg_cpu_usage[1]['854x480'] = 0.0
#avg_cpu_usage[1]['1280x720'] = 0.0
#avg_cpu_usage[2]['640x360'] = 0.0
#avg_cpu_usage[2]['854x480'] = 0.0
#avg_cpu_usage[2]['1280x720'] = 0.0


#sum_count = 0.0
#for j in range(0,3):
#    for n in range(0,3):
#        for i in range(0,27,9):
#            avg_cpu_usage[j][resolutions[n]] += sum(cpu_frame_window_conf[0][(j*3)+i+n])
#            sum_count+= len(cpu_frame_window_conf[0][(j*3)+i+n])
#        avg_cpu_usage[j][resolutions[n]] = avg_cpu_usage[j][resolutions[n]] / sum_count
#        sum_count = 0.0

#print avg_cpu_usage


#N = 3
#ind = np.arange(N)
#width = 0.30

#minAvg = []
#SDAvg = []
#HDAvg = []
#for i in range(0,3):
#    minAvg.append(avg_cpu_usage[i]['640x360'])
#    SDAvg.append(avg_cpu_usage[i]['854x480'])
#    HDAvg.append(avg_cpu_usage[i]['1280x720'])

#mins = plt.plot(minAvg,linestyle='-',marker='^',color='0.50',label='640x460')
#SD = plt.plot(SDAvg,linestyle='-',marker='o',color='0.40',label='854x480')
#HD = plt.plot(HDAvg,linestyle='-',marker='D',color='0.30',label='1280x720')
#plt.ylabel("CPU usage(%)")
#plt.xlabel("Frame Window Size")
#plt.xticks(ind,["50","100","200"])

#plt.legend()

#ax=plt.gca()
#cur_ylim=ax.get_ylim()
#ax.set_ylim([cur_ylim[0],530])

#plt.savefig('../figures/CPU_Usage.png', bbox_inches='tight')


#############################################################################
#### Parse memory results into frame window size / resolution pair averages.#
#############################################################################
frame_window_conf = []

window_count = 0
for scenario in range(0,1):
    frame_window_conf.append([])
    for measure in mem_results[scenario]:
        frame_window_conf[scenario].append([])
        frame_window_conf[scenario][window_count] = measure[10:80]
        frame_window_conf[scenario].append([])
        frame_window_conf[scenario][window_count+1] = measure[100:170]
        frame_window_conf[scenario].append([])
        frame_window_conf[scenario][window_count+2] = measure[200:260]
        window_count += 3

# One dictionary per frame window size.
# Each dictionary contains the average for that resolution/size configuration.
# [50,100,200]
avg_mem_usage = [dict(),dict(),dict()]

avg_mem_usage[0]['640x360'] = 0.0
avg_mem_usage[0]['854x480'] = 0.0
avg_mem_usage[0]['1280x720'] = 0.0
avg_mem_usage[1]['640x360'] = 0.0
avg_mem_usage[1]['854x480'] = 0.0
avg_mem_usage[1]['1280x720'] = 0.0
avg_mem_usage[2]['640x360'] = 0.0
avg_mem_usage[2]['854x480'] = 0.0
avg_mem_usage[2]['1280x720'] = 0.0

retfac_mem_usage = [dict(),dict(),dict()]

retfac_mem_usage[0]['640x360'] = dict()
retfac_mem_usage[0]['854x480'] = dict()
retfac_mem_usage[0]['1280x720'] = dict()
retfac_mem_usage[1]['640x360'] = dict()
retfac_mem_usage[1]['854x480'] = dict()
retfac_mem_usage[1]['1280x720'] = dict()
retfac_mem_usage[2]['640x360'] = dict()
retfac_mem_usage[2]['854x480'] = dict()
retfac_mem_usage[2]['1280x720'] = dict()


# Find averages for memory usage for the different configurations.
sum_count = 0.0
for j in range(0,3):
    for n in range(0,3):
        for i in range(0,27,9):
            print frame_window_conf[0][((j*3)+n+i)]
            retfac_mem_usage[j][resolutions[n]][retargetfactors[i/9]] = \
                    sum(frame_window_conf[0][((j*3)+n+i)]) / len(frame_window_conf[0][((j*3)+n+i)])
            avg_mem_usage[j][resolutions[n]] += sum(frame_window_conf[0][(j*3)+i+n])
            sum_count+= len(frame_window_conf[0][(j*3)+i+n])
        avg_mem_usage[j][resolutions[n]] = avg_mem_usage[j][resolutions[n]] / sum_count
        sum_count = 0.0

#for i in range(0,3):
#    for item in retfac_mem_usage[i]:
#        print retfac_mem_usage[i][item]

avgs = []
for i in range(0,3):
    avgs.append([])
    for j in range(0,3):
        for n in range(0,3):
            avgs[i].append(retfac_mem_usage[n][resolutions[i]][retargetfactors[j]])


print avgs

plt.plot(avgs[0],linestyle='-',marker='s',color='0.50', label='640x360')
plt.plot(avgs[1], linestyle='-',marker='o',color='0.40', label='854x480')
plt.plot(avgs[2], linestyle='-',marker='D',color='0.30', label='1280x720')

plt.ylabel("Memory Usage (%)")
plt.xlabel("Configuration")
plt.xticks(ind,[1,2,3,4,5,6,7,8,9])
plt.legend()

ax=plt.gca()
cur_ylim=ax.get_ylim()
ax.set_ylim([0,15])

plt.savefig('../figures/Memory_distribution.png', bbox_inches='tight')
plt.gcf().clear()

#N = 3
#ind = np.arange(N)
#width = 0.30

#minAvg = []
#SDAvg = []
#HDAvg = []
#for i in range(0,3):
#    minAvg.append(avg_mem_usage[i]['640x360'])
#    SDAvg.append(avg_mem_usage[i]['854x480'])
#    HDAvg.append(avg_mem_usage[i]['1280x720'])

#mins = plt.plot(minAvg,linestyle='-',marker='^',color='0.50',label='640x460')
#SD = plt.plot(SDAvg,linestyle='-',marker='o',color='0.40',label='854x480')
#HD = plt.plot(HDAvg,linestyle='-',marker='D',color='0.30',label='1280x720')
#plt.ylabel("Memory usage(%)")
#plt.xlabel("Frame Window Size")
#plt.xticks(ind,["50","100","200"])

#plt.legend()

#ax=plt.gca()
#cur_ylim=ax.get_ylim()
#ax.set_ylim([cur_ylim[0],15])

#plt.savefig('../figures/Memory_Usage.png', bbox_inches='tight')
#plt.show()


#ax.set_xticks(ind+(width*1.5))
#ax.set_xticklabels(("640x360","854x480","1280x720"))
#plt.plot(mem_results[0][0][10:250])
#plt.plot(mem_results[0][1][10:250])
#plt.plot(mem_results[0][2][10:250])
#plt.ylabel("Memory usage(%)")
#plt.xlabel("Seconds (s)")
#plt.axis([0,250,0,20])
#plt.show()
#print "Battle of averages:"
#for i in range(0,9):
#    print "Min:",minRetLatency[i],"vs.",minPassOneLatency[i]
#    print "SD:",SDRetLatency[i],"vs.",SDPassOneLatency[i]
#    print "HD:",HDRetLatency[i],"vs.",HDPassOneLatency[i]


#print "MIN:"
#for item in minTotalLatency:
#    print item
#print "SD:"
#for item in SDTotalLatency:
#    print item
#print "HD:"
#for item in HDTotalLatency:
#    print item

#print toni_dict
#print rini_dict 
#print plni_dict



#dtype = dict(resolutions = resolutions, results=results)
#array = np.array

#print retargeting_results[0].values()
#
#results = retargeting_results[0].values()

#minInit = initial_latency[0]['640x360']
#SDInit = initial_latency[0]['854x480']
#HDInit = initial_latency[0]['1280x720']

#minFRetarget = first_retargeting_results[0]['640x360']
#SDFRetarget = first_retargeting_results[0]['854x480']
#HDFRetarget = first_retargeting_results[0]['1280x720']

#minReSe = recv_send_latency[0]['640x360']
#SDReSe = recv_send_latency[0]['854x480']
#HDReSe = recv_send_latency[0]['1280x720']

#minReAd = recv_add_latency[0]['640x360']
#SDReAd = recv_add_latency[0]['854x480']
#HDReAd = recv_add_latency[0]['1280x720']

#minQaSe = qadd_send_latency[0]['640x360']
#SDQaSe = qadd_send_latency[0]['854x480']
#HDQaSe = qadd_send_latency[0]['1280x720']

#minQSum = qadd_send_sum[0]['640x360']
#SDQSum = qadd_send_sum[0]['854x480']
#HDQSum = qadd_send_sum[0]['1280x720']

#print "---SUM QUEUE/SEND---\n"
#print "MIN:"
#for item in minQSum:
#    print item
#print "\nSD:"
#for item in SDQSum:
#    print item
#print "\nHD:"
#for item in HDQSum:
#    print item

#print "\n---AVG QUEUE/SEND---\n"
#print "MIN:"
#for item in minQaSe:
#    print item
#print "\nSD:"
#for item in SDQaSe:
#    print item
#print "\nHD:"
#for item in HDQaSe:
#    print item

#print "\n---AVG RECV/ADD---\n"
#print "MIN:"
#for item in minReAd:
#    print item
#print "\nSD:"
#for item in SDReAd:
#    print item
#print "\nHD:"
#for item in HDReAd:
#    print item
#
#print "\n---AVG RECV/SEND---\n"
#print "MIN:"
#for item in minReSe:
#    print item
#print "\nSD:"
#for item in SDReSe:
#    print item
#print "\nHD:"
#for item in HDReSe:
#    print item


#print "MIN:"
#for item in minFRetarget:
#    print item
#print "\nSD:"
#for item in SDFRetarget:
#    print item
#print "\nHD:"
#for item in HDFRetarget:
#    print item

#print "MIN:\n",minInit
#print "SD:\n",SDInit
#print "HD:\n",HDInit


#minRetarget = retargeting_results[0]['640x360']
#SDRetarget = retargeting_results[0]['854x480']
#HDRetarget = retargeting_results[0]['1280x720']

# Scenario 3
#AminRetarget = retargeting_results[1]['640x360']
#ASDRetarget = retargeting_results[1]['854x480']
#AHDRetarget = retargeting_results[1]['1280x720']

#print "\nMIN:"
#for item in AminRetarget:
#    print item
#print "\nSD:"
#for item in ASDRetarget:
#    print item
#print "\nHD:"
#for item in AHDRetarget:
#    print item

#AminInit = initial_latency[1]['640x360']
#ASDInit = initial_latency[1]['854x480']
#AHDInit = initial_latency[1]['1280x720']

#print "\nInitial latency scenario 3"
#print "MIN: \n", AminInit
#print "SD: \n", ASDInit
#print "HD: \n", AHDInit

#print "MIN:\n",minRetarget
#print "SD:\n",SDRetarget
#print "HD:\n",HDRetarget


#fpsPerFactor = []
#c = 0
#for i in range(0,9,3):
#    fpsPerFactor.append([])
#    fpsPerFactor[c].append(minRetarget[i])
#    fpsPerFactor[c].append(SDRetarget[i])
#    fpsPerFactor[c].append(HDRetarget[i])
#    c+=1

#for x in fpsPerFactor:
#    print x

#minTotalLatency = toni_dict['640x360']
#SDTotalLatency = toni_dict['854x480']
#HDTotalLatency = toni_dict['1280x720']
minRetLatency = rini_dict['640x360']
SDRetLatency = rini_dict['854x480']
HDRetLatency = rini_dict['1280x720']
minPassOneLatency = avg_energy_latency[0]['640x360']
#SDPassOneLatency =  avg_energy_latency[0]['858x480']
#HDPassOneLatency = avg_energy_latency[0]['1280x720']

#print avg_energy_latency

#avg_energy_latency[0]['854x480'] = avg_energy_latency.pop('858x480')

#print avg_energy_latency

#plt.show()

############ BAR CHART EXAMPLE

# 640x360 version
N = 9
ind = np.arange(N)
width = 0.30


avg_50_percent = []
avg_100_percent = []
avg_200_percent = []

# Compares the initial latency average against the overall average of first pass + crop + 1seamcarve
for i in range (0,3):
    avg_50_percent.append(0)
    avg_100_percent.append(0)
    avg_200_percent.append(0)
    retLatency = rini_dict[resolutions[i]]
    passOneLatency = avg_energy_latency[0][resolutions[i]]

    for j in range(0,len(retLatency),3):
        avg_50_percent[i] += ((passOneLatency[j] / retLatency[j]) * 100)
        avg_100_percent[i] += ((passOneLatency[j+1] / retLatency[j+1]) * 100)
        avg_200_percent[i] += ((passOneLatency[j+2] / retLatency[j+2]) * 100)
        #print "Percentage:", ((passOneLatency[j] / retLatency[j]) * 100)

    avg_50_percent[i] = avg_50_percent[i] / 3
    avg_100_percent[i] = avg_100_percent[i] / 3
    avg_200_percent[i] = avg_200_percent[i] / 3

    fig, ax = plt.subplots()
    firstBars  = ax.bar(ind+width,retLatency,width,color='#cde6ff')
    retBars = ax.bar(ind,passOneLatency,width,color='#cdffd4')

    ax.set_ylabel("Milliseconds (ms)")
    ax.set_xlabel("Configuration")
    ax.set_xticks(ind+width)
    ax.set_xticklabels(ind)
    ax.set_yticks(np.arange(0,4000,500))

    ax.legend(("Initial latency","Average latency"),prop={'size':13})
    fig.savefig('../figures/Initial_latency_'+resolutions[i]+'.png', bbox_inches='tight')
    #plt.show()

#print avg_50_percent,":",avg_100_percent,":",avg_200_percent

avg_reinit = [0.0,0.0,0.0]

#print reinit_latency[1]

for i in range(0,3):
    for item in reinit_latency[1][resolutions[i]]:
        avg_reinit[i] += item

    avg_reinit[i] = avg_reinit[i] / len(reinit_latency[1][resolutions[i]])
#print avg_reinit

N = 3
ind = np.arange(N)
width = 0.30

fig, ax = plt.subplots()
bars  = ax.bar(ind+width,avg_reinit,width,color='#cde6ff')

ax.set_ylabel("Milliseconds (ms)")
ax.set_xlabel("Resolution")
ax.set_xticks(ind+(width*1.5))
ax.set_xticklabels(("640x360","854x480","1280x720"))
ax.set_yticks(np.arange(0,50,5))

fig.savefig('../figures/Reinit_latency.png',bbox_inches='tight')

#fir, ax = plt.subplots()
#minBars = ax.bar(ind,minRetarget,width,color='g')
#SDBars  = ax.bar(ind+width,SDRetarget,width,color='b')
#HDBars  = ax.bar(ind+(width*2),HDRetarget,width,color='r')

#ax.set_ylabel("Frames per second")
#ax.set_xlabel("Configuration")
#ax.set_title("Retargeting rates for the differing configurations")
#ax.set_xticks(ind+(width*1.5))
#ax.set_xticklabels(ind)

#resolutions = list(retargeting_results[0].keys())

#ax.legend((minBars[0], SDBars[0], HDBars[0]), (resolutions[0],resolutions[2],resolutions[1]))

#plt.show()

#plt.bar(range(len(retargeting_results[0])), retargeting_results[0].values()[0][:3], align='center')
#plt.xticks(range(len(retargeting_results[0])), list(retargeting_results[0].keys()))
#plt.ylabel("Retargeting rate")
#plt.xlabel("Resolution")
#plt.show()

#for dictionary in retargeting_results:
#    print dictionary


#################################################################################
# Relative increase in initial latency between frame window size configurations #
#################################################################################

resolutions = [[],[],[]]

resolutions[0] = [222.6,514.6,1016.9,236.7,436.1,898.7,243.5,453.4,978.7]
resolutions[1] = [416.2,836.4,1545.6,493.4,818.0,1642.4,448.3,804.4,1404.8]
resolutions[2] = [860.5,1222.4,2654.9,865.3,1419.8,3515.8,901.7,1435.9,3274.8]

increase = [[],[],[]]

for i in range(0,3):
    for j in range(0,9):
        if j+1 < 9 and (j+1) % 3 != 0:
            print j," : ", float(resolutions[i][j+1] / resolutions[i][j])
            increase[i].append(float('%.2f'%float(resolutions[i][j+1] / resolutions[i][j])))

print increase

results = [[],[],[]]

N = 6
ind = np.arange(N)
width = 0.25

fig, ax = plt.subplots()
rects1 = ax.bar(ind, increase[0], width, color='#cde6ff')
rects2 = ax.bar(ind+width, increase[1], width, color='#cdffd4')
rects3 = ax.bar(ind+(width*2), increase[2], width, color='#ffcdd2')

ax=plt.gca()
ax.set_ylabel('Times increase')
ax.set_ylim([1,3])
ax.set_xlabel('Configurations')
ax.set_xticks(ind+(width*1.5))
ax.set_xticklabels(('1-2','2-3','4-5','5-6','7-8','8-9'))

ax.legend((rects1[0],rects2[0],rects3[0]),('640x360','854x480','1280x720'))

#plt.show()
fig.savefig('../figures/Initlatency_increase.png',bbox_inches='tight')


