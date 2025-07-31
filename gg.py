K=int(input("Enter the Integer"))
l=[]
for i in range(1,K+1):
    if K%i==0:
        l.append(i)

print(list(l))