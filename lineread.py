c = 0
with open('C://Users//anand//Documents//soclive.txt') as fp:
    f = open("opt.txt","a")
    for line in fp:
        f.write(line)
        c +=1
        if(c==17248450):
            break
