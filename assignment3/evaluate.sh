#!/bin/bash

#Do NOT Terminate terminate with a "/"

current_dir=$(pwd)
INPUT="$current_dir/Evaluation/input"
OUTPUT="$current_dir/Evaluation/output"

touch "A3-marks.txt"
MARKSFILE="$current_dir/A3-marks.txt"

echo "====================START==========================" >> $MARKSFILE
date >> $MARKSFILE

for FOLDER in SUBMIT/*
 do
	#! echo FOLDER NAME : "${FOLDER}"
	cd "${FOLDER}"
	ROLLNO=$(ls *.cu | tail -1 | cut -d'.' -f1)
	echo "$ROLLNO"
	# check for single source file. If not halt script!
	if [ $(ls | wc -l) -ne 1 ] 
	then
		echo "May be cleanup files! (delete all files in 'A3-Submit/<ROLLNO>' folder except ROLLNO.cu) and run evaluate.sh!"
		break
	fi
	#copy stud's code to main.cu
	cp ${ROLLNO}.cu "$current_dir/main.cu"
	
	#create log file for stud
	LOGFILE="$FOLDER/${ROLLNO}.log"
	cd ../..
	nvcc main.cu -arch=sm_70 --std=c++11 -o main.out
	date >> $LOGFILE
	for testcase in $INPUT/*; 
	do
		filename=${testcase##*/}
		echo "$filename"
		echo "$filename" >> $LOGFILE
		./main.out  "$INPUT/$filename"
		diff  "$OUTPUT/$filename" "./output.txt" -b > /dev/null 2>&1
		exit_code=$?
		if (($exit_code==0)); then
			echo "success"
			echo "success" >> $LOGFILE
	
		else
			echo "failure"
			echo "failure" >> $LOGFILE
		fi
	
	done
	SCORE=$(grep -ic success $LOGFILE) #Counts the success in log
	#! TOTAL=$(ls $INPUT/*.txt | wc -l)
	echo $ROLLNO,$SCORE 
	echo $ROLLNO,$SCORE >> $MARKSFILE # write to file
	 
done

date >> $MARKSFILE
echo "====================DONE!==========================" >> $MARKSFILE


