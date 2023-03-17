#A quick script to aggregate data stored in the results folder. change the inputDir/folders list below to aggregate data in a different location

import os
from os.path import exists

#control metadata
folders = ["gqf", "point", "bloom", "sqf", "rsqf"]
sizes = ["22", "24", "26", "28", "30", "32"]
inputDir = "results"

ops = ["insert", "exists-lookup", "false-lookup"]

for folder in folders:
    
    for op in ops:
        
        output = inputDir + "/aggregate/" + folder + "-aggregate-" + op + ".txt"
        with open(output, "w") as output_file:

            output_file.write("x_0 y_0\n")
            for size in sizes:
                #double check this is consistent with the new standard format
                filename = inputDir + "/" + folder + "/" + size + "-" + op + ".txt"
                
                #aggregates are split over 20 batches and are nitems/second
                
                five_percent = .05* 2**float(size)
                #print(five_percent)

                print(filename)
                if exists(filename):
                    with open(filename, "r") as input_file:
                        lines = input_file.readlines()
                        if len(lines) > 0:
                            #clip first line - always junk
                            lines = lines [1:]
                            temp_sum = 0
                            for line in lines:

                                #extract just the value
                                #temp_sum += five_percent / float(line.split(" ")[-1]) 


                                temp_sum += five_percent/float(line.split(" ")[-1]) 
                                #temp_sum +=  1/float(line.split(" ")[-1])
                            print(temp_sum)
                            
                            inverted = (2**float(size))/temp_sum
                            print("inverted: {}".format(inverted))

                            output_file.write("{} {}\n".format(size, inverted))
                            #output_file.write("{} {}\n".format(size, 1/temp_sum))

