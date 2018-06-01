import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


######### Determinar el centro gemetrico de una imagen de camara AllSky

### Calculo del centro geometrico de una imagen de cielo completo usando N rectas, para lo cual se requieren 2N puntos.
### Esto lo hace calculando la interseccion entre rectas perpendiculares a las N rectas introducidas.

"""
### Recibe los puntos con los cuales se forman las N rectas secantes.
print ("Numero de rectas a utilizar")
N = int( raw_input('N='))
x= []
y= []

if (N==1):
    print ("Se requieren minimo dos rectas")

else:    
    for i in range(N):
        print ("Puntos de la recta %d"%i)
        x0= float( raw_input('x0='))
        y0= float( raw_input('y0='))
        x.append(x0)
        y.append(y0)
        print ("\n")
         
        x1= float( raw_input('x1='))
        y1= float( raw_input('y1='))
        print ("\n")
       
        x.append(x1)
        y.append(y1)

## Imagen Berrio
N=3
x=[169.476,30.130,22.665,211.778,35.107,59.990]
y=[1027.617,808.645,301.028,14.871,826.063,208.960]
"""

### Numero de rectas y coordenadas x,y de los puntos que las conforman
N=6
x=[155.400,132.936,186.504,801.672,41.352,153.672,894.984,922.632,805.128,936.456,181.320,872.520]
y=[639.128,70.616,666.776,663.320,392.024,44.696,540.632,215.768,23.960,276.704,11.864,105.176]

### Obtiene las ecuaciones de las rectas perpendiculares a las rectas introducidas manualmente, con pendiente -1/m y que pase por el punto medio de los regmentos entre los bordes de la imagen del AllSky

eq_x = []
eq_y = []
eq_sol = []
for i in range(N):
    
    xi = (x[(2*i)+1]-x[2*i])
    yi = (y[(2*i)+1]-y[2*i])

    xm = (x[2*i+1]+x[2*i])/2
    ym = (y[2*i+1]+y[2*i])/2
    
    if ((xi == 0 and yi!=0)):
        
        eq_x.append(0)
        eq_y.append(1)
        eq_sol.append(ym)
        #print "hola1"
    else:
        if (yi==0 and xi!=0):
            eq_x.append(1)
            eq_y.append(0)
            eq_sol.append(xm)
            #print "hola2"
        else:
            m = yi/xi
            
            eq_x.append(-1/m)
            eq_y.append(1)
            eq_sol.append(((xm/m)+ym))
 

### Calcular las soluciones a la interseccion de las N rectas
x_sol=[]
y_sol=[]

for i in range(N):
    for j in range(i+1,N):
        
        a = np.array([ [eq_x[i],eq_y[i]],[eq_x[j],eq_y[j]]])
        b = np.array([eq_sol[i],eq_sol[j]])
        solution= abs(np.linalg.solve(a,b))

        x_sol.append(solution[0])
        y_sol.append(solution[1])



### Calculo de los promedios de las intersecciones de las rectas, para definir con mayor precision el centro geometrico
sumax=0
for x in x_sol:
    sumax+=x


sumay=0
for y in y_sol:
    sumay+=y

xc=sumax/len(x_sol)
yc=sumay/len(y_sol)
#print ("centro: x=%f, y=%f"%(xc,yc))


##################################################################

### Calcular la distancia entre puntos en la imagen y el centro geometrico
"""
print ("################################")
print ("Numero de puntos")
Np = int( raw_input('Np='))
x_star= []
y_star= []

for i in range(Np):
    print ("Coordenadas del punto %d"%i)
    x0= float( raw_input('x%d='%i))
    y0= float( raw_input('y%d='%i))
    x_star.append(x0)
    y_star.append(y0)
    print ("\n")    
"""

R=[]
Rx=[]

#Coordenadas estrellas imagen meteoro Medellin (meteor.fit): alpha centauri, beta centauri jupiter, luna, antares, saturno, marte, arturo, spica, 14 crb, eta Her
x_star=[836.232,851.784,628.872,485.448,621.960,509.640,551.112,429.360,687.624,362.760,326.472] #coordenada x en la imagen
y_star=[191.576,217.496,303.896,46.424,196.760,63.704,119.000,502.616,468.056,409.304,371.288]   #coordenada y en la imagen
az=[182.376,187.127,16.107,116.679,146.770,120.275,127.703,321.231,232.888,30.481,40.316]         #coordenada de azimut
h=[2.795,2.509,65.059,27.101,50.611,30.018,40.096,73.612,61.126,62.166,5.012]                    #coordenada de altura

Ri=210
Z=[]
Rf=[]
Theta=[]

#Calculo de las componentes del vector posicion
for i in range(len(x_star)):
    rx=( x_star[i] - xc)
    ry=(y_star[i] - yc)
    
    
    r= np.sqrt(rx**2 + ry**2)
    
    theta = np.arctan2(ry,rx)*(180/np.pi)
    if (theta < 0):
        theta = 360-abs(theta)

    Theta.append(theta)
    R.append(r)
    #print r
    """
    if (r> Ri-deltar and r<= Ri+deltar):
        #print x_star[i], y_star[i]
        xdelta.append(x_star[i])
        ydelta.append(y_star[i])
         
    """
        #z=az[i]*(np.pi/180)
    z=  (np.pi*0.5)- h[i]*(np.pi/180)
    Rf.append(r)
    Z.append(z)
        

###########################
# Bolido (Coordenadas iniciales y finales)


x_b0 = 391.823                                    #Coordenada x imagen
y_b0 = 366.146                                    #Coordenada y imagen
r_b0 = ( (x_b0-xc)**2 +  (y_b0-yc)**2)**0.5       #Distancia inicial al centro geometrico
z_b0 = np.arcsin(r_b0/Ri)                         #Distancia Cenital inicial  
h_b0 = 90.0 - z_b0*(180.0/np.pi)                  #Altura inicial (grados)
theta_b0 = np.arctan2(y_b0,x_b0)*(180/np.pi)
if (theta_b0 < 0):
    theta_b0 = 360-abs(theta_b0)


x_bf = 386.833                                    #Coordenada x imagen
y_bf = 370.000                                    #Coordenada y imagen
r_bf = ( (x_bf-xc)**2 +  (y_bf-yc)**2)**0.5       #Distancia final al centro geometrico
z_bf = np.arcsin(r_bf/Ri)                         #Distancia cenital final 
h_bf = 90.0 - z_bf*(180.0/np.pi)                  #Altura final (grados)
theta_bf = np.arctan2(y_bf,x_bf)*(180/np.pi)

if (theta_bf < 0):
    theta_bf = 360-abs(theta_bf)


t=10          #Tiempo de exposicion (s)

#############################3
#Curva de azimuth vs theta, y determinacion de la ecuacion de la recta seguida por los puntos

y1=187.127000
x1=341.050675
y0=30.481000
x0=152.014256
xp=285.367988
yp=127.70300

m= (y1-y0) / (x1-x0)

x=np.linspace(0,360,100)
y= m * x + (y1- m*x1)

A0=m * theta_b0 + (y1- m*x1)   #Azimut inicial
Af=m * theta_bf + (y1- m*x1)   #Azimut final


if (A0 < 0):
    A0 = 360-abs(A0)

if (Af < 0):
    Af = 360-abs(Af)

dh = h_bf - h_b0   #delta de altura
dA = Af - A0       #delta de azimut

###### Great-circle distance: Calculo de la distancia angular recorrida por el bolido

delta = 2*np.arcsin(( np.sin(dh*0.5)**2 + np.cos(h_b0)*np.cos(h_bf)*np.sin(dA*0.5)**2 )**0.5)
""" 
print ("A0=%f"%A0)
print ("Af=%f"%Af)
print ("h0=%f"%h_b0)
print ("hf=%f"%h_bf)
print ("Angular Distance=%f\n"%delta)
print ("######################################\n")
"""

##### Estimacion de la velocidad minima del meteoro

R = np.linspace(0,100,10)  #Diferentes alturas propuestas para el bolido
d = R*delta                #Distancia lineal recorrida para cada altura

v= d/t                     #Velocidad para cada altura

for i in range(len(v)):
    print ("Height (km)= %f, Velocity (km/s) =%f"%(R[i],v[i]))


#######################################################
#Graficas

plt.figure(1)
plt.plot(Theta,az,'.',label="Datos")
plt.plot(x,y,'r',label="Line recta")
plt.legend()
plt.xlabel("$\\theta$")
plt.ylabel("$A$")
plt.xlim(0,360)
plt.ylim(0,360)
plt.grid()
plt.title("Angulo azimutal vs $\\theta$")
plt.savefig("angles.png")
#plt.show()

"""
for i in range(len(az)):
    print ("Theta=%f" %Theta[i])
    print ("A=%f\n" %az[i])
    
"""

"""
plt.figure(1)
plt.plot (xdelta,ydelta,'.',xc,yc,'r*')
plt.show()
"""
"""
plt.figure(1)
plt.plot(Z,Rx,'.'
plt.show()

def function(Z,f):              
    return f*np.sin(Z)                                                                                                                             
                                                                                                                                                     
popt,pcov =curve_fit(function,Z,f,[310])
""" 
x=np.linspace(0,2,100) 
y=[Ri*np.sin(a) for a in x]

plt.figure(4)
plt.plot(Z,Ri*np.sin(Z),'k.')
plt.plot(x,y)                                                                                                       
plt.xlabel('Z')
plt.ylabel('r')
plt.title('r = R sinZ ')
plt.grid()
plt.savefig('calibracion.png')

plt.figure(0)
plt.plot(x_star,y_star, ".",xc,yc,"r*",x_b0,y_b0,'g.',x_bf,y_bf,'y.')
plt.xlabel("X(pix)")
plt.ylabel("Y(pix)")
plt.grid()
plt.savefig("stars.png")


##############################################
#De coordenadas horizontales a cartesianas geocentricas (eje x hacia el punto vernal)

TSL = 133.65*(np.pi/180)                   #Tiempo sideral local Marzo 11, 2018. 3:45 am
lat = 6.21694444444*(np.pi/180)            #Latitud Medellin 
h = h_bf* (np.pi/180)                      #Altura en radianes
A = Af*(np.pi/180)                         #Azimut en radianes


RA = TSL - H                                                                       #Ascencion recta
DEC = np.arcsin( np.sin(lat)*np.sin(h) + np.cos(lat)*np.cos(h)*np.cos(A)     )     #Declinacion
H = np.arccos( (np.sin(h) - np.sin(dec)*np.sin(lat))/ (np.cos(dec)*np.cos(lat)) )  #Angulo horario

r_earth =  6370                      #Radio de la tierra
r= r_earth + min(R)                  #Distancia del centro de la tierra al bolido

x = r*np.cos(RA)*np.cos(DEC)
y = r*np.sin(RA)*np.cos(DEC)
z = r*np.sin(DEC)
