
def msToS(value):
    return (value / 1000)

input_data_dir = "raw_data/"
output_data_dir = "parsed_data/"

for scenario in range(2,6):
#for scenario in range(4,5):
    for retarget in range (1,4):
        for framewinsize in range(1,4):
            filename = str(scenario) + "_" + str(retarget) + "_" + str(framewinsize) + ".txt"
            outfilename = str(filename).split('.')[0]
            outfilename = outfilename + "_Parsed.txt"
            print outfilename

            recv_measures = []
            add_measures = []
            qadd_measures = []
            send_measures = []

            recv_count = []
            add_count = []
            qadd_count = []
            send_count = []
            recv_count.append(0)
            add_count.append(0)
            qadd_count.append(0)
            send_count.append(0)

            reconfiguration_count = 0

            diff_addsend = []
            diff_add = []
            diff_send = []

            diff_recv_avg = []
            diff_send_avg = []
            diff_add_avg = []
            sum_add = []
            sum_qadd = []

            diff_recv_avg.append(0)
            diff_send_avg.append(0)
            diff_add_avg.append(0)
            sum_add.append(0)
            sum_qadd.append(0)

            diff_first_recv_avg = []
            diff_first_send_avg = []
            diff_first_add_avg = []
            sum_first_add = []
            sum_first_qadd = []

            diff_first_recv_avg.append(0)
            diff_first_send_avg.append(0)
            diff_first_add_avg.append(0)
            sum_first_add.append(0)
            sum_first_qadd.append(0)

            configuration_time = []
            configuration_time.append(0)

            reinitialization_time = []
            reinitialization_time.append(0)

            input_resolution = []
            output_resolution = []
            frame_size = []
            retargeting_factor = []

            total_time = []
            first_window = []
            energy = []
            avg_energy = []
            crop = []
            trans = []
            seam_carving = []
            frames = []


            #input_file = open(str(sys.argv[1]),'r')
            input_file = open(input_data_dir+filename,'r')
            outfile = open(output_data_dir+outfilename,'w')

            for line in input_file:
                if line[:4] == "SEND":
                    send_measures.append(line[5:])
                    send_count[reconfiguration_count] += 1
                elif line[:4] == "RECV":
                    recv_measures.append(line[5:])
                    recv_count[reconfiguration_count] += 1
                elif line[:4] == "QADD":
                    qadd_measures.append(line[5:])
                    qadd_count[reconfiguration_count-1] += 1
                elif line[:3] == "ADD":
                    add_measures.append(line[4:])
                    add_count[reconfiguration_count] += 1
                elif line[:10] == "Input size":
                    input_resolution.append(line[12:].strip())
                elif line[:11] == "Target size":
                    output_resolution.append(line[13:].strip())
                elif line[:14] == "Retargetfactor":
                    retargeting_factor.append(line[16:].strip())
                elif line[:17] == "Frame window size":
                    frame_size.append(int(line[18:].strip()))
                elif line[:7] == "RESULTS":
                    values = line.split()
                    total_time.append(float(values[1]))
                    energy.append(float(values[2]))
                    crop.append(float(values[3]))
                    trans.append(float(values[4]))
                    seam_carving.append(float(values[5]))
                    first_window.append(float(values[6]))
                    frames.append(int(values[7]))
                    avg_energy.append(0.0)
                    # Average first pass + crop + 1 frame in seam carve
                    avg_energy[reconfiguration_count] = \
                        (energy[reconfiguration_count] + crop[reconfiguration_count] + \
                        (seam_carving[reconfiguration_count] / float(frames[reconfiguration_count]))) / \
                       (float(frames[reconfiguration_count]) / float(frame_size[reconfiguration_count]))

                    reconfiguration_count += 1
                    # Count for next configuration.
                    recv_count.append(0)
                    add_count.append(0)
                    qadd_count.append(0)
                    send_count.append(0)

            send_index = 0
            recv_index = 0
            add_index = 0
            qadd_index = 0

            for i in range(0,reconfiguration_count):
                #print "SEND COUNT:",send_count[i],"RECV_COUNT:",recv_count[i],"QADD_COUNT:",qadd_count[i],"ADD_COUNT:",add_count[i]
                for j in range(0,send_count[i]):
                    diff_addsend.append((int(send_measures[send_index + j]) - int(add_measures[add_index + j])))
                    diff_add.append((int(add_measures[add_index + j]) - int(recv_measures[recv_index + j])))
                    diff_send.append((int(send_measures[send_index + j]) - int(qadd_measures[qadd_index + j])))
                    #if diff_send[len(diff_send)-1] != 0 and qadd_index+j < 100:
                    #    print "Q:",qadd_index+j,":",qadd_measures[qadd_index+j]
                    #    print "S:",send_index+j,":",send_measures[send_index+j]

                send_index = send_index + send_count[i]
                recv_index = recv_index + recv_count[i]
                qadd_index = qadd_index + qadd_count[i]
                add_index = add_index + add_count[i]

            segment_count = 0
            current_buffer = 0

            send_index = 0
            recv_index = 0
            add_index = 0
            qadd_index = 0

            # Calculates the average latency for each retargeting configuration.
            # Disregards the first frame window in each. 
            for i in range(0,reconfiguration_count):
                for j in range(0,send_count[segment_count]):
                    if j > int(frame_size[segment_count]):
                        diff_recv_avg[segment_count] += diff_addsend[current_buffer]
                        diff_add_avg[segment_count] += diff_add[current_buffer]
                        diff_send_avg[segment_count] += diff_send[current_buffer]
                        sum_add[segment_count] += diff_add[current_buffer]
                        sum_qadd[segment_count] += diff_send[current_buffer]
                    else:
                        diff_first_recv_avg[segment_count] += diff_addsend[current_buffer]
                        diff_first_add_avg[segment_count] += diff_add[current_buffer]
                        sum_first_add[segment_count] += diff_add[current_buffer]
                        if diff_send[current_buffer] > -1:
                            diff_first_send_avg[segment_count] += diff_send[current_buffer]
                        if diff_send[current_buffer] > -1:
                            sum_first_qadd[segment_count] += diff_send[current_buffer]


                    current_buffer += 1
                segment_count += 1
                diff_recv_avg.append(0)
                diff_send_avg.append(0)
                diff_add_avg.append(0)
                sum_add.append(0)
                sum_qadd.append(0)
                diff_first_recv_avg.append(0)
                diff_first_send_avg.append(0)
                diff_first_add_avg.append(0)
                sum_first_add.append(0)
                sum_first_qadd.append(0)

            configuration_start = 0
            recv_start = 0

            # Average difference for each configuration.
            for i in range(0,reconfiguration_count):
                diff_recv_avg[i] = float(diff_recv_avg[i]) / float(send_count[i])
                diff_add_avg[i] = float(diff_add_avg[i]) / float(send_count[i])
                diff_send_avg[i] = float(diff_send_avg[i]) / float(send_count[i])
                diff_first_recv_avg[i] = float(diff_first_recv_avg[i]) / float(frame_size[i])
                diff_first_add_avg[i] = float(diff_first_add_avg[i]) / float(frame_size[i])
                diff_first_send_avg[i] = float(diff_first_send_avg[i]) / float(frame_size[i])

                configuration_time[i] = int(send_measures[configuration_start + send_count[i]-1]) - int(recv_measures[recv_start])
                configuration_time.append(0)
                if i > 0:
                    reinitialization_time[i-1] = int(recv_measures[recv_start+1]) - int(send_measures[configuration_start-1]) 
                    reinitialization_time.append(0)
                configuration_start += send_count[i]
                recv_start += recv_count[i]


            current_buffer = 0
            sum_smoothing = 0

            # Format for parsed file (each resolution configuration):
            # - Resolution
            # - Retargeting factor
            # - Frame window size
            # - Retargeting rate average
            # - Retargeting rate average first window
            # - Transitional smoothing time
            # - Initial latency
            # - Average difference recv/send
            # - Average difference recv/add
            # - Average difference qadd/send
            # - Sum time recv/add
            # - Sum time qadd/send
            # - Initial retargeting latency
            # - Reinitialization time
            # - Average latency first pass
            # - Total retargeting time
            
            #print "\n--------- Results ---------\n"
            if scenario < 4 or scenario == 5:
                for i in range (0, reconfiguration_count):
                    #print "\n:::: Configuration ",i+1,"::::"
                    #print "--------------------------------"
                    outfile.write(input_resolution[i]+" ")
                    outfile.write(retargeting_factor[i]+ " ")
                    outfile.write(str(frame_size[i])+" ")
                    if total_time[i] > 0:
                        outfile.write(str((frames[i] / msToS(total_time[i])))+" ")
                    else:
                        outfile.write("0 ");
                    outfile.write(str((frame_size[i] / msToS(first_window[i])))+" ")
                    outfile.write(str(trans[i])+" ")
                    outfile.write(str(diff_addsend[current_buffer])+" ")
                    outfile.write(str(diff_recv_avg[i])+" ")
                    outfile.write(str(diff_add_avg[i])+" ")
                    outfile.write(str(diff_send_avg[i])+" ")
                    outfile.write(str(sum_add[i])+" ")
                    outfile.write(str(sum_qadd[i])+" ")
                    outfile.write(str(first_window[i])+" ")
                    if i > 0:
                        outfile.write(str(reinitialization_time[i-1])+" ")
                    else:
                        outfile.write("0 ");
                    outfile.write(str(avg_energy[i])+" ")

                    current_buffer += send_count[i]

                    if total_time[i] > 0:
                        outfile.write(str(total_time[i])+" ")
                    outfile.write("\n");
            else:

                avg_ret_init = [0,0,0]
                avg_total_init = [0,0,0]
                avg_plugin_impact = [0,0,0]
                for i in range(0, reconfiguration_count):
                    if input_resolution[i] == "640x360":
                        avg_ret_init[0] += first_window[i]
                        avg_total_init[0] += diff_addsend[current_buffer]
                        avg_plugin_impact[0] += (diff_addsend[current_buffer] - first_window[i])
                    elif input_resolution[i] == "854x480":
                        avg_ret_init[1] += first_window[i]
                        avg_total_init[1] += diff_addsend[current_buffer]
                        avg_plugin_impact[1] += (diff_addsend[current_buffer] - first_window[i])
                    else:
                        avg_ret_init[2] += first_window[i]
                        avg_total_init[2] += diff_addsend[current_buffer]
                        avg_plugin_impact[2] += (diff_addsend[current_buffer] - first_window[i])


                    current_buffer += send_count[i]
                for i in range(0,3):
                    avg_ret_init[i] = avg_ret_init[i] / 10.0
                    avg_total_init[i] = avg_total_init[i] / 10.0
                    avg_plugin_impact[i] = avg_plugin_impact[i] / 10.0

                # Format for initial latency scenario:
                # - Average total initial latency
                # - Average retargeting initial latency
                # - Average plugin impact

                current_buffer = 0

                #for i in range(0, reconfiguration_count):
                for i in range(0, 3):
                    outfile.write(input_resolution[i]+" ")
                    outfile.write(str(avg_total_init[i])+" ")
                    outfile.write(str(avg_ret_init[i])+" ")
                    outfile.write(str(avg_plugin_impact[i])+" ")

                    #current_buffer += send_count[i]
                    outfile.write("\n");
                    
                    #print "Resolution: ", input_resolution[i], "->", output_resolution[i]
                    #print "Retargeting factor: ", ((1 - float(retargeting_factor[i])) * 100),"%"
                    #print "Frame window size: ", frame_size[i]
                    #print "Total frames:", frames[i]
                    #print "- without first window:", frames[i] - frame_size[i]
                    #print "Total retargeting time:", (total_time[i] + first_window[i]),"msec"
                    #print "- first frame window took:", first_window[i],"msec"
                    #print "- second window took:",total_time[i],"msec"
                    #print "Total configuration time:", configuration_time[i],"msec"
                    #sum_smoothing += trans[i]
                    #print "Retargeting rate first window:", (frame_size[i] / msToS(first_window[i])), "frames per second."

                    #if total_time[i] > 0:
                    #    print "Retargeting rate overall:", (frames[i] / msToS(total_time[i])), "frames per second."
                    #print "Transitional smoothing time:", trans[i]
                    #print "\n+++++++ Latency ++++++++"
                    #print "Initial latency induced:", diff_addsend[current_buffer],"msec"
                    #print "Average difference between receive and send:", diff_recv_avg[i],"msec"
                    #print "Average difference between receive and add:", diff_add_avg[i],"msec"
                    #print "Average difference between queue add and send:", diff_send_avg[i],"msec"
                    #print "Sum time between recv and add:",sum_add[i],"msec"
                    #print "Sum time between qadd and send:",sum_qadd[i],"msec"
                    #print "First Average difference between receive and send:", diff_first_recv_avg[i],"msec"
                    #print "First Average difference between receive and add:", diff_first_add_avg[i],"msec"
                    #print "First Average difference between queue add and send:", diff_first_send_avg[i],"msec"
                    #print "First Sum time between recv and add:",sum_first_add[i],"msec"
                    #print "First Sum time between qadd and send:",sum_first_qadd[i],"msec"
                    #if i > 0:
                    #    print "Reinitialization time:",reinitialization_time[i-1]

                    #current_buffer += send_count[i]
                    #outfile.write("\n");
            #print "Total transitional smoothing time: ", sum_smoothing

