import pandas as pd

def main():
    x,y = list((4,4,4,5)), list();
    y.append(2)
    
    print(max(set(x), key=lambda f: x.count(f)))
    print(x)

    for i in range(5,35,5):
        print(i)

if __name__ == "__main__":
    main()