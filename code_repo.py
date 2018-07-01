#Function Graph Example
#d is the input function, eg:
def d(n):
    return n ** 2

import numpy as np
import matplotlib.pyplot as plt


t = np.arange(0, 127, 1)
plt.plot(t, list(map(d, t)))
plt.ylabel('YLabel')
plt.xlabel('XLabel')
plt.title('Title')
plt.show()

#Histogram Example
#takes input of a list of numbers and a number of bins, eg:
sample = [1]* 10 + [2]* 5 + [6]*3
bins = 5
plt.hist(sample, bins) 
plt.ylabel('Ylabel')
plt.xlabel('Xlabel')
plt.title('Title')
plt.show()

#Log Log plot creator 

import numpy as np
import matplotlib.pyplot as plt
def d(n): #being the function to be plotted
    return n ** 8 + 800 * n



t = np.arange(1, 127, 1)
logt = []
logfunction = []
for a in t:
    logt.append(np.log10(a))
    logfunction.append(np.log10(d(a)))
plt.plot(logt, logfunction)
plt.ylabel('YLabel')
plt.xlabel('XLabel')
plt.title('Title')
plt.show()
#print(logt)
#print(logfunction)

import numpy as np
import matplotlib.pyplot as plt

def Legendre(N,x):
    
    P = np.zeros((N+1,len(x)))
    Q = np.zeros((N+1,len(x)))
    
    P[0], P[1] = 1 + 0*x, x
    Q[0], Q[1] = 0, 1
    
    for i in range(2, N + 1):
        P[i] = (2*i - 1)/(i) * x * P[i-1] -(i - 1)/(i)*P[i-2]
        Q[i] =(2*i - 1)*P[i - 1] + Q[i - 2]
        
    return P, Q

i = np.linspace(-1, 1, 2000)
Pns, Qns = Legendre(100, i)
plt.figure(0)
plt.subplot(211)
for x in range(10):
    plt.plot(i, Pns[x], label = "$P$" + str(x))
plt.plot(i, Pns[100], label = "$P$" + str(100))
plt.legend(bbox_to_anchor = (1.05, 1))
plt.title("Legendre Polynomials")
plt.ylabel('$y$')
plt.xlabel('$x$')
plt.grid(True)
plt.legend


plt.subplot(212)
for x in range(10):
    plt.plot(i, Qns[x], label = "$Q$" + str(x))
plt.legend(bbox_to_anchor = (1.05, 1))
plt.title("Derivatives of Legendre Polynomials")
plt.ylabel('$y$')
plt.xlabel('$x$')
plt.grid(True)
plt.subplots_adjust(top=3, bottom=0, left=0, right=2, wspace=0.5)


plt.show()



def T(x): 
    return 256 * x ** 9 - 576 * x ** 7 + 432 * x ** 5 - 120 * x ** 3 + 9 * x
def U(x): 
    return 2304 * x ** 8 - 4032 * x ** 6 + 2160 * x ** 4 - 360 * x ** 2 + 9
xs = np.linspace(-1, 1, 2000)
ys1 = T(xs)
ys2 = U(xs)
plt.figure(0)
plt.subplot(121)
plt.plot(xs, ys1)
plt.title("$T(x)$")
plt.ylabel('$T(x)$')
plt.xlabel('$x$')

plt.subplot(122)
plt.plot(xs, ys2)
plt.title("$U(x)$")
plt.ylabel('$U(x)$')
plt.xlabel('$x$')


plt.subplots_adjust(top=1, bottom=0, left=0, right=2, wspace=0.5)

plt.show()