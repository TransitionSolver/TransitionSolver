#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt 
import math




#在引力波计算参考文献中已知的参数：
vw = 0.67            #Vw--bubble wall的速度
g = 106.75          #the Standard Model value g∗ 
T = 3115             #transition temperature T∗  为什么成核温度的差别会这么大呢？？？
b = 0.2             #H∗/β--the phase transition duration ???
c = 0.09           #α--the phase transition strength
k_v = 0.44
k_turb = 0.044

x = np.linspace(10**(-5),1,10**5)
y1 = []
y2 = []
y3 = [] 

#计算Acoustic waves的贡献：
for f in x:
    def f_sw():
        f_sw = 1.9 * 10**(-5) * (1/vw)*(1/b)*(T/100)*(g/100)**(1/6) 
        return f_sw
    
    def h2O_sw():
        h2O_sw= 2.65*10**(-6)*b**2*(k_v*c/(1+c))**2*(100/g)**(1/3)*vw*(f/f_sw())**3*(7/(4+3*(f/f_sw())**2))**(7/2)
        return h2O_sw
    #the gravitational wave power spectrum
    
#计算 Turbulence的贡献：
    def f_turb():
        f_turb = 2.7 * 10**(-5) * (1/vw) * (1/b) * (T/100) * (g/100)**(1/6)
        return f_turb
        
    def h(): #1705.01783
        h = 16.5*10**(-6)*(T/100)*(g/100)**(1/6)
        return h
    
    def h2O_turb():
        h2O_turb = 3.35*10**(-4)*b**2*((k_turb*c)/(1+c))**(3/2)*(100/g)**(1/3)*vw*((f/f_turb())**3)/(((1+(f/f_turb()))**(11/3)) * (1+(8*math.pi*f/h())))
        return h2O_turb
    #the gravitational wave power spectrum
    
#总贡献= Acoustic waves的贡献+ Turbulence的贡献：
    def h2O_total():
        h2O_total = h2O_sw() + h2O_turb()
        return h2O_total

   
    y1.append(h2O_sw())
    y2.append(h2O_turb())
    y3.append(h2O_total())
    
a1 = np.loadtxt(r'/home/xuzhongxiu/装置曲线/C1.txt') #导入gws.txt中的数据 
x_01 = a1[:,0]  #读取第一列所有数据
y_41 = a1[:,1]  #读取第二列所有数据

a2 = np.loadtxt(r'/home/xuzhongxiu/装置曲线/C2.txt') #导入gws.txt中的数据 
x_02 = a2[:,0]  #读取第一列所有数据
y_42 = a2[:,1]  #读取第二列所有数据

a3 = np.loadtxt(r'/home/xuzhongxiu/装置曲线/C3.txt') #导入gws.txt中的数据 
x_03 = a3[:,0]  #读取第一列所有数据
y_43 = a3[:,1]  #读取第二列所有数据

a4 = np.loadtxt(r'/home/xuzhongxiu/装置曲线/C4.txt') #导入gws.txt中的数据 
x_04 = a4[:,0]  #读取第一列所有数据
y_44 = a4[:,1]  #读取第二列所有数据

a5= np.loadtxt(r'/home/xuzhongxiu/装置曲线/Ulyimate-DECIGO.txt') #导入gws.txt中的数据 
x_05 = a5[:,0]  #读取第一列所有数据
y_45 = a5[:,1]  #读取第二列所有数据

a6= np.loadtxt(r'/home/xuzhongxiu/装置曲线/ALIA.txt') #导入gws.txt中的数据 
x_06 = a6[:,0]  #读取第一列所有数据
y_46 = a6[:,1]  #读取第二列所有数据

a7= np.loadtxt(r'/home/xuzhongxiu/装置曲线/DECIGO.txt') #导入gws.txt中的数据 
x_07 = a7[:,0]  #读取第一列所有数据
y_47 = a7[:,1]  #读取第二列所有数据

a8= np.loadtxt(r'/home/xuzhongxiu/装置曲线/BBO.txt') #导入gws.txt中的数据 
x_08 = a8[:,0]  #读取第一列所有数据
y_48 = a8[:,1]  #读取第二列所有数据



fig,ax=plt.subplots(1,1,figsize=(8,6),dpi=180,facecolor='white')
plt.plot(x,y1[:],color='blue',linestyle='--')
plt.plot(x,y2[:],color='black',linestyle='--')
plt.plot(x,y3[:],color='red')

plt.plot(x_01,y_41[:],color='red',linewidth= 0.7)
plt.plot(x_02,y_42[:],color='blue',linewidth= 0.7)
plt.plot(x_03,y_43[:],color='orange',linewidth= 0.7)
plt.plot(x_04,y_44[:],color='green',linewidth= 0.7)
plt.plot(x_05,y_45[:],color='purple',linewidth= 0.7)
plt.plot(x_06,y_46[:],color='cyan',linewidth= 0.7)
plt.plot(x_07,y_47[:],color='yellow',linewidth= 0.7)
plt.plot(x_08,y_48[:],color='black',linewidth= 0.7)


plt.fill_between(x_01,y_41,10**(-4),facecolor = 'red', alpha = 0.2) #设置LISA sensitivity阴影区域
plt.fill_between(x_02,y_42,10**(-4),facecolor = 'blue', alpha = 0.2)
plt.fill_between(x_03,y_43,10**(-4),facecolor = 'orange', alpha = 0.2)
plt.fill_between(x_04,y_44,10**(-4),facecolor = 'green', alpha = 0.2)
plt.fill_between(x_05,y_45,10**(-4),facecolor = 'purple', alpha = 0.2)
plt.fill_between(x_06,y_46,10**(-4),facecolor = 'cyan', alpha = 0.2)
plt.fill_between(x_07,y_47,10**(-4),facecolor = 'yellow', alpha = 0.2)
plt.fill_between(x_08,y_48,10**(-4),facecolor = 'black', alpha = 0.2)

#plt.tick_params(axis='x',colors='black')
#plt.tick_params(axis='y',colors='black')
plt.xscale('log') #设置横坐标的缩放
plt.yscale('log') #设置纵坐标的缩放
plt.xlim(10**(-5), 1)
plt.ylim(10**(-22), 10**(-4))

plt.xlabel('f(HZ)')
plt.ylabel('h^2ΩGW(f)')
plt.legend(['sound waves','turbulence','All','C1','C2','C3','C4','Ulyimate-DECIGO','ALIA','DECIGO','BBO'],bbox_to_anchor=(1.05, 0.54), loc=3, borderaxespad=0) #图例 
plt.show()




