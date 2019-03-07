lines=list(open('output.txt','r'))
# print(lines)
# lines=lines.sor
lines = [a.split(" ") for a in lines]
b=sorted(lines)
outfile=open("output.txt",'w')
for line in b:
    outfile.write(line[0] + " " + line[1] + " " + line[2])
outfile.close()
