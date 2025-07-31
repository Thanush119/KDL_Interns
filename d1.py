def gdg(n,g):
    for i in range(n):
        s1="  "*(n-i-1)
        s2="  "*(2*i+g)
        print(s1+"R"+s2+"G")

    for i in range(n-1,-1,-1):  
        s1="  "*(n-i-1)                                    
        s2="  "*(2*i+g)
        print(s1+"B"+s2+"Y")

gdg(5,4)

print(2*"\n")
def print_gdg_large(n):
    for i in range(n):
        for letter in "GDG":  
            for j in range(n):
                if letter == "G":
                    if (i == 0 and j != 0) or (i == n-1 and j != 0) or (j == 0 and i != 0 and i != n-1) or (j == n-1 and i >= n//2) or (i == n//2 and j >= n//2):
                        print("*", end=" ")
                    else:
                        print(" ", end=" ")
                elif letter == "D":
                    if j == 0 or (i == 0 and j != n-1) or (i == n-1 and j != n-1) or (j == n-1 and i != 0 and i != n-1):
                        print("*", end=" ")
                    else:
                        print(" ", end=" ")
            print("  ", end="")  
        print()

print_gdg_large(7)