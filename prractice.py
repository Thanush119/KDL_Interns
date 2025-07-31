def conv(n):
    rem=list()
    while n/2!=1:
        x=n%2
        rem.insert(x)
    rem.insert(1)
    print(rem)

conv(5)
