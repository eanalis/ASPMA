def sineModelMultiRes(x,fs,w1,w2,w3,N1,N2,N3,B1,B2,B3,t):
    '''Analysis/synthesis of a sound using the sinusoidal model with multiple window sizes and without sine tracking
	x: input array sound
    w1, w2, w3: three analysis windows with different sizes
    N1, N2, N3: size of complex spectrum to be used with w2,w3,w3 respectively
    B1,B2,B3: the high range edges of frequency bands to used to select peak frequency from three DFTs
    t: threshold in negative dB 
    returns y:output array sound'''	
    hM11 = int(math.floor((w1.size+1)/2))               #half first analysis window size by rounding
    hM12 = int(math.floor((w1.size)/2))                 #half first analysis window size by floor
    hM21 = int(math.floor((w2.size+1)/2))               #half second analysis window size by rounding 
    hM22 = int(math.floor((w2.size)/2))                 #half second analysis window size by floor 
    hM31 = int(math.floor((w3.size+1)/2))               #half third analysis window size by rounding 
    hM32 = int(math.floor((w3.size)/2))                 #half third analysis window size by floor 
    Ns = 512                                            #Synthesis FFT size
    H = Ns//4                                           #Hop size used for analysis and synthesis
    hNs = Ns//2                                         #Half synthesis fft size
    pin = max(hNs,hM11,hM21,hM31)                       #initialize frame pointer in the middle of the largest analysis window
    pend = x.size - max(hNs,hM11,hM21,hM31)             #last sample to start a frame is length of audio minus half largest analysis window
    yw = np.zeros(Ns)                                   #initialize output sound frame
    y = np.zeros(x.size)                                #initialize output aray
    w1 = w1/sum(w1)                                     #Normalize analysis window
    w2 = w2/sum(w2)                                     #Normalize analysis window
    w3 = w3/sum(w3)                                     #Normalize analysis window
    sw = np.zeros(Ns)                                   #initialize synthesis window
    ow = triang(2*H)                                    #triangular window
    sw[hNs-H:hNs+H] = ow                                #add triangular window
    bh = blackmanharris(Ns)                             #blackman harris window
    bh = bh /sum(bh)                                    #normalized blackman harris window   
    sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H] #normalized synthesis window
    while pin<=pend:
#----analysis----
        ipfreq = []                                     #array to hold selected frequencies from the three DFTs
        ipmag = []                                      #array to hold selected magnitudes from the three DFTs
        ipphase = []                                    #array to hold selected phases from the three DFTs
        x1 = x[pin-hM11:pin+hM12]                       #frame for the first DFT
        x2 = x[pin-hM21:pin+hM22]                       #frame for the second DFT
        x3 = x[pin-hM31:pin+hM32]                       #frame for the third DFT
        mX1,pX1 = DFT.dftAnal(x1,w1,N1)                 #magnitude and phase spectrum from first DFT
        mX2,pX2 = DFT.dftAnal(x2,w2,N2)                 #magnitude and phase spectrum from second DFT
        mX3,pX3 = DFT.dftAnal(x3,w3,N3)                 #magnitude and phase spectrum from third DFT
        ploc1 = UF.peakDetection(mX1,t)                 #find peaks in the spectrum from first DFT
        ploc2 = UF.peakDetection(mX2,t)                 #find peaks in the spectrum from second DFT
        ploc3 = UF.peakDetection(mX3,t)                 #find peaks in the spectrum from third DFT
        iploc1,ipmag1,ipphase1 = UF.peakInterp(mX1,pX1,ploc1)
        for i in range(len(iploc1)):                    #iterate over 1st peaks and add to the frequency array if they are within first band, then add corresponding magnitudes and phases
            ipfreq1 = fs * iploc1[i] / float(N1)
            if ipfreq1 < B1:
                ipfreq.append(ipfreq1)
                ipmag.append(ipmag1[i])
                ipphase.append(ipphase1[i])                                 
        iploc2,ipmag2,ipphase2 = UF.peakInterp(mX2,pX2,ploc2)
        for i in range(len(iploc2)):                   #iterate over 2nd peaks and add to the frequency array if they are within second band, then add corresponding magnitudes and phases
            ipfreq2 = fs * iploc2[i] / float(N2)
            if (ipfreq2 >= B1 and ipfreq2 < B2) :
                ipfreq.append(ipfreq2)
                ipmag.append(ipmag2[i])
                ipphase.append(ipphase2[i])
        iploc3,ipmag3,ipphase3 = UF.peakInterp(mX3,pX3,ploc3)
        for i in range(len(iploc3)):                   #iterate over 3rd peaks and add to the frequency array if they are within third band, then add corresponding magnitudes and phases
            ipfreq3 = fs * iploc3[i] / float(N3)
            if (ipfreq3 >= B2 and ipfreq3 < B3) :
                ipfreq.append(ipfreq3)
                ipmag.append(ipmag3[i])
                ipphase.append(ipphase3[i])
        ipfreq = np.array(ipfreq)                       #convert to numpy array
        ipmag = np.array(ipmag)                         #convert to numpy array
        ipphase = np.array(ipphase)                     #convert to numpy array
#-----Synthesis-----
        Y = UF.genSpecSines(ipfreq,ipmag,ipphase,Ns,fs) #generate sines in the spectrum
        fftbuffer = np.real(ifft(Y))                    #compute inverse fft
        yw[:hNs-1] = fftbuffer[hNs+1:]                  #undo zero-phase window
        yw[hNs-1:] = fftbuffer[:hNs+1]                  
        y[pin-hNs:pin+hNs] += sw*yw                     #overlap, add and apply synthesis window
        pin += H                                        #advance sound pointer
    return y