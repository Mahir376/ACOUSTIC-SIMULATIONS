import numpy as np
from scipy.fft import fft,ifft

def stft(insig:np.ndarray, winsize:int, fftsize:int|None=None) -> np.ndarray:
    """
    STFT
    I
    insig (np.ndarray): input signal, shape [T,C]
    winsize (int): window size
    fftsize (int, def. None): fft transform length. If not provided, fftsize = 2*winsize
    O
    spectrum: STFT of $insig with the given parameters
    """
    if insig.ndim == 1:
        insig = insig.reshape(-1, 1)
    lSig = insig.shape[0]
    nCHin = insig.shape[1]
    # time-frequency processing
    if fftsize==None:
        fftsize = 2*winsize
    hopsize = winsize//2
    nBins = fftsize//2 + 1
    nWindows = int(np.ceil(lSig/winsize))
    nFrames = 2*nWindows+1
    # zero pad the signal's start and end for STFT
    #pad in time dimension
    insig_pad = np.pad(insig, ((winsize//2,(nFrames*winsize//2-lSig)), (0,0) ))
    #insig_pad = [zeros(winsize/2, nCHin); insig; zeros(nFrames*winsize/2-lSig, nCHin)];
    spectrum = np.zeros((nBins, nFrames, nCHin),dtype=np.complex64)
    # transform window (hanning)
    x = np.array(range(winsize))
    win = np.sin(x*(np.pi/winsize))**2
    win = np.expand_dims(win,1) #add extra dimension to match input
    # processing loop
    idx = 0
    nf = 0
    while nf < nFrames:
        # Window input and transform to frequency domain
        insig_win = (win * np.ones((1,nCHin)))* insig_pad[idx:winsize+idx,:]
        inspec = fft(insig_win, fftsize, axis=0) #by default first axis with dim !=1 is used in matlab
        inspec = inspec[0:nBins,:] # keep up to nyquist
        spectrum[:,nf,:] = inspec
        # advance sample pointer
        idx = idx + hopsize
        nf = nf + 1
    return spectrum


def ctf_ltv_direct(sig: np.ndarray, irs: np.ndarray, ir_times: np.ndarray, fs:int, winsize:int, print_progress:bool=False) -> np.ndarray:
    """
    I
    sig (numpy.ndarray): input signal, shape (T) or (T,C)
    irs (numpy.ndarray): array containing impulse responses, shape (T,Nrirs) or (T,C,Nrirs)
    ir_times (numpy.ndarray): time instant for each ir, shape (Nrirs)
    fs (int): sampling rate
    winsize (int): analysis / synthesis window size for STFT
    print_progress (bool, def. False): whether to print export progress
    O
    convsig (np.ndarray): interpolated convolutions of $sig and $irs at $ir_times
    """
    #STFT
    hopsize = winsize//2
    fftsize = 2*winsize
    nBins = fftsize//2+1
    #IR
    lIr = irs.shape[0]
    if len(irs.shape)==2:
        nIrs = irs.shape[1]
        nCHir = 1
    elif len(irs.shape)==3:
        nIrs = irs.shape[2]
        nCHir = irs.shape[1]
    else:
        raise Exception("Unsupported number of dimensions for input irs")
    if nIrs != ir_times.shape[-1]:
        raise Exception("nIrs != ir_times.shape[-1]. Check inputs")
    if len(sig.shape)==1:
        sig = np.expand_dims(sig,-1)
        #print("Added channel dimension to input ( (T)->(T,1) )")
    # number of STFT frames for the IRs (half-window hopsize)
    nIrWindows = int(np.ceil(lIr/winsize)) #cast to int
    nIrFrames = 2*nIrWindows+1
    # number of STFT frames for the signal (half-window hopsize)
    lSig = sig.shape[0]
    nSigWindows = int(np.ceil(lSig/winsize))
    nSigFrames = 2*nSigWindows+1
    #nCHsig = sig.shape[1] #<-- commented because unused
    # quantize the timestamps of each IR to multiples of STFT frames (hopsizes)
    tStamps = np.round((ir_times*fs+hopsize)/hopsize).astype(np.int32)
    nIntFrames = tStamps[-1]
    tStamps = tStamps - 1 #fix for python indexing :)
    # create the two linear interpolator tracks, for the pairs of IRs between
    # timestamps
    Gint = np.zeros((nIntFrames, nIrs))
    for ni in range(nIrs-1):
        #ni=1:nIrs-1
        #in matlab the second index is inclusive,  tStamps(ni):tStamps(ni+1); -> tStamps[ni:ni+2]
        tpts = tStamps[ni:ni+2]
        tpts = np.arange(tpts[0],tpts[-1]+1)
        ntpts = len(tpts)
        Gint[tpts,ni] = 1 - np.array(range(ntpts))/(ntpts-1)
        Gint[tpts,ni+1] = np.array(range(ntpts))/(ntpts-1)
    if nCHir==1:
        irspec = np.zeros((nBins, nIrFrames, nIrs),dtype=np.complex64)
    else:
        irspec = np.zeros((nBins, nIrFrames, nCHir, nIrs),dtype=np.complex64)
    for ni in range(nIrs):
        if nCHir ==1:
            irspec[:, :, ni] = stft(irs[:, ni], winsize)[:, :, 0]
        else:
            irspec[:,:,:,ni] = stft(irs[:,:,ni], winsize)
    #compute input signal spectra
    sigspec = stft(sig, winsize)
    #initialize interpolated time-variant ctf
    Gbuf = np.zeros((nIrFrames,nIrs))
    if nCHir == 1:
        ctf_ltv = np.zeros((nBins,nIrFrames),dtype=np.complex64)
    else:
        ctf_ltv = np.zeros((nBins,nIrFrames,nCHir),dtype=np.complex64)
    S = np.zeros((nBins, nIrFrames),dtype=np.complex64)

    nFrames = int(np.min((nSigFrames, nIntFrames)))
    # processing loop
    idx = 0
    nf = 0
    convsig = np.zeros( (winsize//2 + nFrames*winsize//2 + fftsize - winsize, nCHir) )
    #inspec_pad = [inspec zeros(nBins,nIrFrames)];
    inspec_pad = sigspec
    while nf < nFrames:
        # compute interpolated ctf
        Gbuf[1:] = Gbuf[:-1]
        Gbuf[0] = Gint[nf]
        if nCHir ==1:
            for nif in range(nIrFrames):
                ctf_ltv[:,nif] = np.dot(np.squeeze(irspec[:,nif,:]), Gbuf[nif,:].T)
        else:
            for nch in range(nCHir):
                for nif in range(nIrFrames):
                    #ctf_ltv[:, nif, nch] = irspec[:, nif, nch, :] @ Gbuf[nif]
                    #or
                    ctf_ltv[:,nif,nch] = np.dot(np.squeeze(irspec[:,nif,nch,:]),(Gbuf[nif,:].T)) #np.dot(np.squeeze(irspec[:,nif,nch,:]),Gbuf[nif,:].T)

        inspec_nf = inspec_pad[:,nf]
        S[:,1:nIrFrames] = S[:,0:nIrFrames-1]
        S[:,0] = np.squeeze(inspec_nf)
        convspec_nf = np.squeeze( np.sum(  np.tile(np.expand_dims(S,axis=-1),[1,1,nCHir]) * ctf_ltv, axis=1) )
        convspec_nf = np.concatenate( (convspec_nf, np.conj(convspec_nf[-2:0:-1, :])), axis=0)
        convsig_nf = np.real(ifft(convspec_nf,fftsize,axis=0))
        # overlap-add synthesis
        convsig[idx:idx+fftsize,:] = convsig[idx:idx+fftsize,:] + convsig_nf
        # advance sample pointer
        idx = idx + hopsize
        nf = nf + 1
        if print_progress:
            print(f"{int(np.round(nf/nFrames*100))} %")
    convsig = convsig[winsize:nFrames*winsize//2,:]
    return convsig
