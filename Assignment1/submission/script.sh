for file in sample_test/*.jpg
do
    #echo $file;
    A="$(cut -d'/' -f2 <<<"$file")";
    #echo $A;
    B="$(cut -d'.' -f1 <<<"$A")";
    #echo $B;
    outfile="output"/"$B"+".txt"
    echo $outfile

done;
