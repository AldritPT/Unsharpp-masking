from base64 import b16encode
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math, random
from scipy import ndimage

def escalarGrisesHDTV(image):
    R = image[:,:,2]
    G = image[:,:,1]
    B = image[:,:,0]
    imgG = R*0.2126 + G*0.7152 + B*0.0722
    imgG = imgG.astype(np.uint8)
    return imgG

def escalarGrisesNTSC(image):
    R = image[:,:,2]
    G = image[:,:,1]
    B = image[:,:,0]
    imgG = R*0.299 + G*0.587 + B*0.114
    imgG = imgG.astype(np.uint8)
    return imgG

def escalarGrisesHDR(image):
    R = image[:,:,2]
    G = image[:,:,1]
    B = image[:,:,0]
    imgG = R*0.2627 + G*0.6780 + B*0.0593
    imgG = imgG.astype(np.uint8)
    return imgG

def escalaGrises(image):
    R = image[:,:,2]
    G = image[:,:,1]
    B = image[:,:,0]
    imgG = R*0.33 + G*0.33 + B*0.33
    imgG = imgG.astype(np.uint8)
    return imgG


def normalizar(img):

    image = (2-1) / (img.max()-img.min()) * (img-img.min())
    print("Imagen normalizada")
    print(image)
    return image

def contrasteLog(image):
    c = 255/(np.log10(1+np.max(image)))
    img = c * np.log10(1+image)
    img = np.array(img, dtype = np.uint8)
    
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')
    plt.show()

    plt.imshow(img, cmap='gray')
    plt.title('Imagen contrasteLog:')
    plt.show()

    return img

def contrasteExp(image, gamma):
    gamma_corrected = np.array(255*(image / 255) ** gamma, dtype = 'uint8')

    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')
    plt.show()

    plt.imshow(gamma_corrected, cmap='gray')
    plt.title('Imagen con gamma: '+str(gamma))
    plt.show()

    

def contrasteExpA(image):
    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')
    plt.show()
    for gamma in [0.1, 0.25, 0.5, 0.75, 1.2, 1.5, 1.75, 2.2]:
        
        # Aplicacion de corrección de gamma
        gamma_corrected = np.array(255*(image / 255) ** gamma, dtype = 'uint8')
    
        plt.imshow(gamma_corrected, cmap='gray')
        plt.title('Imagen con gamma:'+str(gamma))
        plt.show()

def negativo(image):
    img_corrected = 255 - image

    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')
    plt.show()

    plt.imshow(img_corrected, cmap='gray')
    plt.title('Imagen negativa')
    plt.show()
    return img_corrected

def neg(image):
    img_corrected = 255 - image
    return img_corrected

def contrasteP(image, a, b):
    img_corrected = (b - a) *(((image - image.min())*255)/(image.max()-image.min())) + a
    
    plt.imshow(img_corrected, cmap='gray')
    plt.title('Imagen corregida')
    plt.show()


def histograma(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    return hist


def rotacion(image, angulo):
    ancho = image.shape[1] #columnas
    alto = image.shape[0] # filas
    # Rotación
    M = cv2.getRotationMatrix2D((ancho//2,alto//2),-angulo,1)
    imageOut = cv2.warpAffine(image,M,(ancho,alto))

    plt.imshow(image, cmap='gray')
    plt.title('Imagen Original')
    plt.show()

    plt.imshow(imageOut, cmap='gray')
    plt.title('Imagen rotada')
    plt.show()

    imageOut = np.array(imageOut, dtype = np.uint8)

    plt.imshow(imageOut[195:220, 245:270], cmap=plt.cm.gray)
    plt.show()

    return imageOut

def NN_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=round((i+1)*(scrH/dstH))
            scry=round((j+1)*(scrW/dstW))
            retimg[i,j]=img[scrx-1,scry-1]
    retimg = np.array(retimg, dtype = np.uint8)
    return retimg

def BiLinear_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    img=np.pad(img,((0,1),(0,1),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=(i+1)*(scrH/dstH)-1
            scry=(j+1)*(scrW/dstW)-1
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            retimg[i,j]=(1-u)*(1-v)*img[x,y]+u*(1-v)*img[x+1,y]+(1-u)*v*img[x,y+1]+u*v*img[x+1,y+1]
    retimg = np.array(retimg, dtype = np.uint8)
    return retimg

def BiBubic(x):
    x=abs(x)
    if x<=1:
        return 1-2*(x**2)+(x**3)
    elif x<2:
        return 4-8*x+5*(x**2)-(x**3)
    else:
        return 0

def BiCubic_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    #img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=i*(scrH/dstH)
            scry=j*(scrW/dstW)
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            tmp=0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if x+ii<0 or y+jj<0 or x+ii>=scrH or y+jj>=scrW:
                        continue
                    tmp+=img[x+ii,y+jj]*BiBubic(ii-u)*BiBubic(jj-v)
            retimg[i,j]=np.clip(tmp,0,255)
    retimg = np.array(retimg, dtype = np.uint8)
    return retimg


def fakeImage(image, image2):
    ig = escalaGrises(image)
    ig2 = escalaGrises(image2)

    plt.imshow(ig, cmap='gray')
    plt.title('Imagen original')
    plt.show()

    plt.imshow(ig2, cmap='gray')
    plt.title('Imagen alterada')
    plt.show()

    ancho = ig.shape[1] #columnas
    alto = ig.shape[0] # filas
    for x in range(-3,4):
        for y in range(-3,4):
            m = np.float32([[1,0,x],[0,1,y]])
            imgT = cv2.warpAffine(ig,m,(ancho,alto))
            res = cv2.subtract(ig2,imgT)
            t = (f'traslacion (x,y) = ({x},{y})')
            plt.title(t)
            plt.imshow(res, cmap='gray')
            plt.figure()
            plt.show

def fakeImage2(image, image2, x, y):
    ig = escalaGrises(image)
    th, bw = cv2.threshold(ig, 127, 255, cv2.THRESH_BINARY)
    ig2 = escalaGrises(image2)
    th2, bw2 = cv2.threshold(ig2, 127, 255, cv2.THRESH_BINARY)

    plt.imshow(ig, cmap='gray')
    plt.title('Imagen original')
    plt.show()

    plt.imshow(ig2, cmap='gray')
    plt.title('Imagen alterada')
    plt.show()

    ancho = bw.shape[1] #columnas
    alto = bw.shape[0] # filas
    m = np.float32([[1,0,x],[0,1,y]])
    imgT = cv2.warpAffine(bw,m,(ancho,alto))
    res = cv2.subtract(bw2,imgT)
    t = (f'traslacion (x,y) = ({x},{y})')
    plt.title(t)
    plt.imshow(res, cmap='gray')
    plt.figure()
    plt.show


def zeroPadding(image):
    t = image.size
    print('Tamaño de imagen original:')
    print(t)
    ig = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0)
    t2 = ig.size
    print('Tamaño de imagen con Zero Padding:')
    print(t2)
    plt.imshow(ig, cmap='gray')
    plt.figure()
    plt.show


def miConvo(M1,nf,nc):
    #Convolución
    #nf número de filas de la imagen
    #nc número de columnas de la imgen
    M2 = np.zeros((nf, nc))
    H = (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    y = 0
    for f in range(1,nf):
        for c in range (1,nc): 
            #print('f: ', f,'c: ', c)   
            sum = 0
            for ff in range(-1,1):
                for cc in range(-1,1):
                    #print('f+ff+1: ', f+ff+1,'c+cc+1: ', c+cc+1)   
                    if (c+cc+1)<nc:
                        if (f+ff+1)<nf:
                            y = M1[f+ff+1,c+cc+1]*H[ff+2,cc+2]
                            #print(y)
                    sum=sum+y
    #Imagen a la que se aplicó la convolución
            M2[f,c]=sum
    return M2

def miConvo2(M1,nf,nc,H):
    #Convolución
    #nf número de filas de la imagen
    #nc número de columnas de la imgen
    M2 = np.zeros((nf, nc))
    H = (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    y = 0
    for f in range(1,nf):
        for c in range (1,nc): 
            #print('f: ', f,'c: ', c)   
            sum = 0
            for ff in range(-1,1):
                for cc in range(-1,1):
                    #print('f+ff+1: ', f+ff+1,'c+cc+1: ', c+cc+1)   
                    if (c+cc+1)<nc:
                        if (f+ff+1)<nf:
                            y = M1[f+ff+1,c+cc+1]*H[ff+2,cc+2]
                            #print(y)
                    sum=sum+y
    #Imagen a la que se aplicó la convolución
            M2[f,c]=sum
    return M2

def miConvoM(M1,H):
    #Convolución
    #nf número de filas de la imagen
    #nc número de columnas de la imgen
    nf,nc=M1.shape
    M2 = np.zeros((nf, nc))
    y = 0
    for f in range(1,nf):
        for c in range (1,nc): 
            #print('f: ', f,'c: ', c)   
            sum = 0
            for ff in range(-1,1):
                for cc in range(-1,1):
                    #print('f+ff+1: ', f+ff+1,'c+cc+1: ', c+cc+1)   
                    if (c+cc+1)<nc:
                        if (f+ff+1)<nf:
                            y = M1[f+ff+1,c+cc+1]*H[ff+2,cc+2]
                            #print(y)
                    sum=sum+y
    #Imagen a la que se aplicó la convolución
            M2[f,c]=sum
    return M2


def ruidoimprueba(a,p,imin,imax):
    #Añade ruido impulsivo a una imagen
    #b=ruidoimp(a,p,imin,imax)
    #b: imagen de salida con ruido
    #a: imagen de entrada
    #p: probabilidad del ruido
    #imin: valor del impulso de ruido mínimo
    #imax: valor del impulso de ruido máximo
    #ceil: función de aproximación hacia arriba 2.65 aproxima a 3
    #[m,n]=a.size
    m = a.shape[1] #columnas
    n = a.shape[0] # filas
    a= np.double(a)
    n=np.ceil(p*m*n)
    n = np.uint(n)
    turno=0
    b=a
    for i in range(1,n):
        k=np.ceil(m*random())
        l=np.ceil(n*random())
        k = np.uint(k)
        l = np.uint(l)
        if turno==0:
            turno=1
            if k<b.shape[0]:
                if l<b.shape[1]:
                    #print('k:',k,'l:',l,'b.shape[0]:',b.shape[0],'b.shape[1]:',b.shape[1])
                    b[k,l]=imax
        else:
            turno=0
            if k<b.shape[0]:
                if l<b.shape[1]:
                    #print('k:',k,'l:',l,'b.shape[0]:',b.shape[0],'b.shape[1]:',b.shape[1])
                    b[k,l]=imin
    b=np.uint8(b)

    return b

def ruidoimp(a,p,imin,imax):
    #Añade ruido impulsivo a una imagen
    #b=ruidoimp(a,p,imin,imax)
    #b: imagen de salida con ruido
    #a: imagen de entrada
    #p: probabilidad del ruido
    #imin: valor del impulso de ruido mínimo
    #imax: valor del impulso de ruido máximo
    #ceil: función de aproximación hacia arriba 2.65 aproxima a 3
    #[m,n]=a.size
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(2, 2, 1)
    plt.imshow(a,cmap="gray")
    plt.title("Imagen original")
    m,n= a.shape
    np= math.ceil((p/10)*m*n)
    turn=0
    b=a
    for i in range(1,np):
        x,y = (m*random.random()),(n*random.random())
        k = round(x)-1
        l = round(y)-1
        if turn==0:
            turn=1
            b[k,l]=imax
        else:
            turn=0
            b[k,l]=imin
   
    fig.add_subplot(2, 2, 2)
    plt.imshow(b,cmap="gray")
    t = (f'Imagen convolucionada P={p}0%')
    plt.title(t)
    plt.show()

def ruidoimp2(a,p,imin,imax):
    #Añade ruido impulsivo a una imagen
    #b=ruidoimp(a,p,imin,imax)
    #b: imagen de salida con ruido
    #a: imagen de entrada
    #p: probabilidad del ruido
    #imin: valor del impulso de ruido mínimo
    #imax: valor del impulso de ruido máximo
    #ceil: función de aproximación hacia arriba 2.65 aproxima a 3
    #[m,n]=a.size
    m,n= a.shape
    np= math.ceil((p/10)*m*n)
    turn=0
    b=a
    for i in range(1,np):
        x,y = (m*random.random()),(n*random.random())
        k = round(x)-1
        l = round(y)-1
        if turn==0:
            turn=1
            b[k,l]=imax
        else:
            turn=0
            b[k,l]=imin
    return b


def threshold(img):
    blur = cv2.GaussianBlur(img,(15,15),0)
    ret,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)

    fig = plt.figure(figsize=(15, 10))
    fig.add_subplot(1, 3, 1)
    plt.imshow(img,cmap='gray')
    plt.title('Imagen original')
    fig.add_subplot(1, 3, 2)
    plt.imshow(blur,cmap='gray')
    plt.title('Blur')
    fig.add_subplot(1, 3, 3)
    plt.imshow(thresh,cmap='gray')
    plt.title('Thresholding')
    plt.show()

def gasuss_noise(image, mean, scale):
    
    image2 = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, scale, image2.shape)
    out = image2 + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)

    return out

def mascara(b):
    
    H = np.array([
        [1, b, 1],
        [b, b**2, b],
        [1, b, 1]
        ]) / (b+2)**2

    return H

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z

def threshold2(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img

def sobel(img, threshold):
    '''
    edge detection based on sobel

    Parameters
    ----------
    img : TYPE
        the image input.
    threshold : TYPE
         varies for application [0 255].

    Returns
    -------
    mag : TYPE
        output after edge detection.

    '''
    G_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    G_y = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            v = sum(sum(G_x * img[i:i+3, j:j+3]))  # vertical
            h = sum(sum(G_y * img[i:i+3, j:j+3]))  # horizon
            mag[i+1, j+1] = np.sqrt((v ** 2) + (h ** 2))
            
    for p in range(0, rows):
        for q in range(0, columns):
            if mag[p, q] < threshold:
                mag[p, q] = 0
    return mag

def prewitt(img):
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    img_prewittx = cv2.filter2D(img, -1, kernelx)
    img_prewitty = cv2.filter2D(img, -1, kernely)
    g = cv2.addWeighted(img_prewittx, 1, img_prewitty, 1, 0)
    
    return g

def laplaciano(img, kernel):
    processed_image = cv2.filter2D(img,-1,kernel)

    return processed_image

def roberts(img,kernel):
    processed_image = cv2.filter2D(img,-1,kernel)

    return processed_image


def higBoost(A):
    k = np.array([[0, -1, 0],
                [-1, A+4, -1],
                [0, -1, 0]])

    k2 = np.array([[-1, -1, -1],
                [-1, A+8, -1],
                [-1, -1, -1]])
    
    return k, k2

def unsharp(img, kernel):
    processed_image = cv2.filter2D(img,-1,kernel)

    return processed_image

def laplaciano2(img):
    '''
    edge detection based on sobel

    Parameters
    ----------
    img : TYPE
        the image input.
    threshold : TYPE
         varies for application [0 255].

    Returns
    -------
    mag : TYPE
        output after edge detection.

    '''
    G_x = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    G_y = np.array([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            v = sum(sum(G_x * img[i:i+3, j:j+3]))  # vertical
            h = sum(sum(G_y * img[i:i+3, j:j+3]))  # horizon
            mag[i+1, j+1] = np.sqrt((v ** 2) + (h ** 2))
    return mag

def sobel2(img, threshold, Kx, Ky):
    '''
    edge detection based on sobel

    Parameters
    ----------
    img : TYPE
        the image input.
    threshold : TYPE
         varies for application [0 255].

    Returns
    -------
    mag : TYPE
        output after edge detection.

    '''
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            v = sum(sum(Kx * img[i:i+3, j:j+3]))  # vertical
            h = sum(sum(Ky * img[i:i+3, j:j+3]))  # horizon
            mag[i+1, j+1] = np.sqrt((v ** 2) + (h ** 2))
            
    for p in range(0, rows):
        for q in range(0, columns):
            if mag[p, q] < threshold:
                mag[p, q] = 0
    return mag

def sobel3(img, Kx, Ky):
    '''
    edge detection based on sobel

    Parameters
    ----------
    img : TYPE
        the image input.
    threshold : TYPE
         varies for application [0 255].

    Returns
    -------
    mag : TYPE
        output after edge detection.

    '''
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            v = sum(sum(Kx * img[i:i+3, j:j+3]))  # vertical
            h = sum(sum(Ky * img[i:i+3, j:j+3]))  # horizon
            mag[i+1, j+1] = np.sqrt((v ** 2) + (h ** 2))
            
    return mag

def sobel4(img, Kx, Ky):
    '''
    edge detection based on sobel

    Parameters
    ----------
    img : TYPE
        the image input.
    threshold : TYPE
         varies for application [0 255].

    Returns
    -------
    mag : TYPE
        output after edge detection.

    '''
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 4):
        for j in range(0, columns - 4):
            v = sum(sum(Kx * img[i:i+5, j:j+5]))  # vertical
            h = sum(sum(Ky * img[i:i+5, j:j+5]))  # horizon
            mag[i+1, j+1] = np.sqrt((v ** 2) + (h ** 2))
            
    return mag


